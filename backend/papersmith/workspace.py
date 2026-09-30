"""Workspace engine behind the chat: reading uploads, handling chat messages and writing the paper.

Every operation follows the same pattern so concurrent jobs never lose each other's changes:
read a snapshot, do the slow model work outside the project lock, then apply the result to a fresh copy
inside storage.edit(). Model calls are serialised per project (one GPU), and the writing loop releases
the model between paragraphs so the chat stays responsive while the paper is written.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from . import assistant as A
from . import bibtex, consistency, extract as X, ingest, jobs, planner, provenance, sources, storage, verifier, writer
from . import textutils as tu
from .config import settings
from .llm import BackendError, agent_backend, get_backend, is_local, report_waits
from .prompts import focused_excerpt
from .models import (ClaimType, DataTable, Draft, DraftParagraph, Figure, Flag, Passage, Project, Reference,
                     SourceFile, now_iso)

log = logging.getLogger(__name__)

MAX_FIGURES_PER_FILE = 12

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def llm_lock(pid: str, backend_name: str | None = None):
    """Serialises model calls for a local model (one GPU). API models take requests in parallel, so the chat
    never waits for the writer; their rate limits are handled in the backend."""
    if not is_local(backend_name):
        return nullcontext()
    with _locks_guard:
        return _locks.setdefault(pid, threading.Lock())


def _log(p: Project, action: str, **detail) -> None:
    p.history.append({"at": now_iso(), "action": action, **detail})


def _file(p: Project, fid: str) -> SourceFile | None:
    return next((f for f in p.sources if f.id == fid), None)


def _next_id(prefix: str, ids: list[str]) -> str:
    nums = [int(i[len(prefix):]) for i in ids if i.startswith(prefix) and i[len(prefix):].isdigit()]
    return f"{prefix}{(max(nums) if nums else 0) + 1}"


def processing(p: Project) -> list[SourceFile]:
    return [f for f in p.sources if f.status in ("queued", "reading", "extracting")]


# ================================================================ uploads

def add_files(pid: str, files: list[tuple[str, bytes]], uploaded_by: str = "") -> tuple[list[str], list[str]]:
    """Store uploads and register them. Returns (accepted file IDs, rejection messages)."""
    accepted, rejected = [], []
    with storage.edit(pid) as p:
        for name, data in files:
            name = Path(name or "file").name[:120]
            kind = ingest.kind_of(name)
            if kind == "other":
                rejected.append(f"{name}: " + ("save it as .pptx, .docx or .xlsx first" if name.lower().endswith((".ppt", ".doc", ".xls"))
                                               else f"not a supported type ({ingest.SUPPORTED})"))
                continue
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                rejected.append(f"{name}: larger than {settings.max_upload_mb} MB")
                continue
            fid = _next_id("D", [f.id for f in p.sources])
            stored = f"{fid}{Path(name).suffix.lower()}"
            storage.write_file(pid, f"uploads/{stored}", data)
            role = "data" if kind in ("csv", "xlsx") else "reference" if kind == "bib" else "own"
            p.sources.append(SourceFile(id=fid, filename=name, kind=kind, role=role, stored_as=stored, size=len(data),
                                        uploaded_by=uploaded_by))
            accepted.append(fid)
            _log(p, "file_uploaded", file=fid, name=name, bytes=len(data))
        if accepted or rejected:
            names = ", ".join(_file(p, f).filename for f in accepted)
            p.messages.append(A.ChatMessage(id=A.next_message_id(p), role="user", text=f"Uploaded {names}" if names else "Upload",
                                            author=uploaded_by,
                                            attachments=accepted, data={"handled": True}))
            if rejected:
                A.say(p, "I couldn't take these files: " + "; ".join(rejected) + ".", kind="error")
            if accepted:
                A.say(p, f"Reading {names}. I'll tell you what I find.", kind="progress", data={"files": accepted})
                if p.stage == "welcome":
                    p.stage = "collecting"
    return accepted, rejected


def _guess_role(doc: ingest.ParsedDoc, filename: str) -> str:
    if doc.kind in ("csv", "xlsx"):
        return "data"
    if doc.kind == "bib":
        return "reference"
    if doc.kind == "pdf":
        first = " ".join(t for _, t in doc.passages[:6]).lower()
        published = bool(doc.meta.get("doi")) or re.search(
            r"(©|copyright|elsevier|springer|ieee xplore|arxiv:|published online|received:? .{0,40}accepted|"
            r"journal of|proceedings of the)", first)
        return "reference" if published else "own"
    return "own"


def _reference_for(doc: ingest.ParsedDoc, filename: str, existing: list[str]) -> Reference:
    meta = doc.meta or {}
    title = (meta.get("title") or doc.title or Path(filename).stem).strip()
    authors = (meta.get("authors") or "").strip()
    year = str(meta.get("year") or "")
    surname = re.findall(r"[A-Za-z]+", authors.split(",")[0].split(" and ")[0])[-1:] if authors else []
    first_word = next((w for w in re.findall(r"[A-Za-z]{4,}", title) if w.lower() not in tu.STOPWORDS), "paper")
    base = re.sub(r"\W", "", ((surname[0] if surname else Path(filename).stem[:10]) + year + first_word).lower()) or "ref"
    key, n = base, 2
    while key in existing:
        key, n = f"{base}{n}", n + 1
    fields = {"title": title, "author": authors, "year": year, "doi": meta.get("doi", "")}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return Reference(key=key, title=title, authors=authors, year=year, venue="", raw_bibtex=f"@article{{{key},\n{body}\n}}")


def _remove_file_content(p: Project, fid: str) -> None:
    """Take out what a file contributed (claims supported only by it, its tables, figures, reference)."""
    src = _file(p, fid)
    prefix = f"{fid}."
    keep = []
    for c in p.ledger.claims:
        own = [s for s in c.sources if s.startswith(prefix)]
        if own and len(own) == len(c.sources):
            if p.outline:
                planner.remove_claim(p.outline, c.id)
            continue
        c.sources = [s for s in c.sources if not s.startswith(prefix)]
        keep.append(c)
    p.ledger.claims = keep
    p.ledger.tables = [t for t in p.ledger.tables if t.source != fid]
    gone_figs = [f for f in p.ledger.figures if f.source == fid]
    for f in gone_figs:
        if f.filename:
            storage.remove_file(p.id, f.filename)
    p.ledger.figures = [f for f in p.ledger.figures if f.source != fid]
    if src and src.reference_key:
        still_cited = any(src.reference_key in c.refs for c in p.ledger.claims)
        if not still_cited:
            p.ledger.references = [r for r in p.ledger.references if r.key != src.reference_key]
        src.reference_key = ""


def ingest_file(pid: str, fid: str, backend_name: str | None, use_nli: bool, job: jobs.Job) -> None:
    snap = storage.load(pid)
    src = _file(snap, fid)
    if not src:
        return
    job.message = f"Reading {src.filename}"
    with storage.edit(pid) as p:
        f = _file(p, fid)
        f.status, f.detail = "reading", "reading the file"
    try:
        doc = ingest.parse(src.filename, storage.local_file(pid, f"uploads/{src.stored_as}").read_bytes())
    except ingest.IngestError as exc:
        _file_failed(pid, fid, str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - a broken file must not break the workspace
        log.exception("parsing %s failed", src.filename)
        _file_failed(pid, fid, f"This file could not be read ({type(exc).__name__}).")
        return

    role = src.role if src.role_set else _guess_role(doc, src.filename)
    passages = [Passage(id=f"{fid}.p{i}", source_id=fid, location=loc, text=text) for i, (loc, text) in enumerate(doc.passages, 1)]
    table_passage: dict[str, str] = {}

    with storage.edit(pid) as p:
        f = _file(p, fid)
        _remove_file_content(p, fid)                      # a re-read replaces what the file gave before
        f.role, f.units, f.unit_label = role, doc.units, doc.unit_label  # type: ignore[assignment]
        for k, t in enumerate(doc.tables, 1):
            tid = _next_id("T", [x.id for x in p.ledger.tables])
            p.ledger.tables.append(DataTable(id=tid, caption=t.caption, columns=t.columns, rows=t.rows, source=fid, location=t.location))
            pid_t = f"{fid}.t{k}"
            passages.append(Passage(id=pid_t, source_id=fid, location=f"{t.location}, table", text=ingest.table_as_text(t)))
            table_passage[pid_t] = tid
        images = sorted(doc.images, key=lambda im: len(im.data), reverse=True)[:MAX_FIGURES_PER_FILE]
        for im in images:
            fig_id = _next_id("F", [x.id for x in p.ledger.figures])
            fname = f"{fig_id.lower()}{im.ext if im.ext in ('.png', '.jpg', '.jpeg', '.gif', '.bmp') else '.png'}"
            storage.write_file(pid, fname, im.data)
            p.ledger.figures.append(Figure(id=fig_id, caption=im.caption, filename=fname, source=fid, location=im.location))
        wake = False
        if doc.kind == "bib":
            refs = bibtex.parse(doc.bibtex)
            existing = p.ledger.ref_map()
            for r in refs:
                existing[r.key] = r
            p.ledger.references = list(existing.values())
            f.units, f.unit_label, f.status, f.detail = len(refs), "entry", "ready", ""
            A.say(p, f"I added {len(refs)} references from {f.filename} to your library. I'll cite them only where your "
                     "material or answers attach them to a claim.", kind="summary", data={"file": fid})
            wake = _after_file(p)
        else:
            if role == "reference":
                ref = _reference_for(doc, f.filename, [r.key for r in p.ledger.references])
                p.ledger.references.append(ref)
                f.reference_key = ref.key
            f.passages = len(passages)
            f.tables, f.figures = len(doc.tables), len(images)
            f.status, f.detail = "extracting", "finding the claims"
            ref_key = f.reference_key
    if doc.kind == "bib":
        if wake:
            start_chat(pid, None, use_nli)
        return
    sources.remove_source(pid, fid)
    sources.add(pid, passages)

    if doc.kind == "image":
        _file_done(pid, fid, [], 0, 0, role, "")
        return

    # tables become claims row by row without a model; prose goes to the model
    row_claims: list[X.Checked] = []
    if role != "reference":
        for k, t in enumerate(doc.tables, 1):
            row_claims += X.table_row_claims(t.caption, t.columns, t.rows, f"{fid}.t{k}")
    readable = [ps for ps in passages if len(ps.text) > 30 and ps.id not in table_passage]
    res = X.ExtractionResult()
    if readable:
        try:
            backend = get_backend(backend_name, role="read")

            def progress(i: int, n: int) -> None:
                job.progress, job.total = i, n
                job.message = f"Reading {src.filename}: part {i} of {n}"

            def waiting(seconds: float) -> None:
                job.message = f"Reading {src.filename}: waiting {round(seconds)} s for the model's rate limit"

            with llm_lock(pid), report_waits(waiting):
                res = X.extract(backend, src.filename, role, readable, on_progress=progress, use_nli=use_nli)
        except BackendError as exc:
            hint = ("Start PaperSmith with PaperSmithAI.bat, then choose Retry on the file." if is_local(backend_name)
                    else "Choose Retry on the file in a little while.")
            _file_failed(pid, fid, f"I read the file but the model could not process it ({exc}). {hint}", keep_text=True)
            return
    _file_done(pid, fid, res.accepted + row_claims, len(res.rejected), res.failed_chunks, role, ref_key, table_passage)


def _file_failed(pid: str, fid: str, message: str, keep_text: bool = False) -> None:
    with storage.edit(pid) as p:
        f = _file(p, fid)
        if not f:
            return
        f.status, f.detail = "error", message
        A.say(p, f"{f.filename}: {message}", kind="error", options=["Retry " + f.filename] if keep_text else [],
              data={"file": fid})
        _log(p, "file_failed", file=fid, detail=message)
        wake = _after_file(p)
    if wake:
        start_chat(pid, None, True)


def _file_done(pid: str, fid: str, found: list[X.Checked], rejected: int, failed_chunks: int, role: str,
               ref_key: str, table_passage: dict[str, str] | None = None) -> None:
    with storage.edit(pid) as p:
        f = _file(p, fid)
        if not f:
            return
        absorbed: list[tuple[str, str]] = []
        added, merged = X.merge_into(p.ledger, found, origin="file", refs=[ref_key] if ref_key else None,
                                     nli=True, absorbed=absorbed)
        placed = _swap_absorbed(p, absorbed)
        figs_by_loc = {(fg.source, fg.location): fg.id for fg in p.ledger.figures}
        for c in added:
            for s in c.sources:
                if table_passage and s in table_passage and table_passage[s] not in c.tables:
                    c.tables.append(table_passage[s])
            if p.outline and c.id not in placed:
                planner.place_claim(p.outline, p.ledger, c)
        # figures on the same slide or page as a claim's passage support it
        passages = {ps.id: ps for ps in sources.load(pid) if ps.source_id == fid}
        for c in added:
            for s in c.sources:
                ps = passages.get(s)
                if ps:
                    fig = figs_by_loc.get((fid, ps.location.split(" (")[0]))
                    if fig and fig not in c.figures:
                        c.figures.append(fig)
        if p.outline:
            planner.refresh_abstract(p.outline, p.ledger)
        f.claims = sum(1 for c in p.ledger.claims if any(s.startswith(f"{fid}.") for s in c.sources))
        f.status, f.detail = "ready", ""
        _log(p, "file_read", file=fid, claims=len(added), merged=merged, rejected=rejected, role=role)

        counts: dict[str, int] = {}
        for c in added:
            counts[c.type.value] = counts.get(c.type.value, 0) + 1
        unit = f"{f.units} {f.unit_label}{'' if f.units == 1 else 's'}" if f.units else ""
        extras = []
        if f.tables:
            extras.append(f"{f.tables} table{'s' if f.tables != 1 else ''}")
        if f.figures:
            extras.append(f"{f.figures} figure{'s' if f.figures != 1 else ''}")
        text = f"Read {f.filename}" + (f" ({unit})" if unit else "") + ": "
        parts = [f"{len(added)} fact{'s' if len(added) != 1 else ''}" if added else "no facts found"]
        parts += extras
        text += ", ".join(parts) + "."
        if failed_chunks:
            text += f" {failed_chunks} part{'s' if failed_chunks != 1 else ''} couldn't be read."
        options = []
        if f.kind == "pdf" and not f.role_set:
            if role == "reference":
                text += f" Looks like a published paper; I'll cite it as [{ref_key}]."
                options.append(f"{f.filename} is my own work")
            else:
                options.append(f"{f.filename} is a paper to cite")
        A.say(p, text, kind="summary", options=options, data={"file": fid, "added": [c.id for c in added]})
        wake = _after_file(p)
        has_draft = p.draft is not None
    if has_draft:
        start_write(pid, None, True)      # new material slots into the paper that is already written
    if wake:
        start_chat(pid, None, True)


def set_role(pid: str, fid: str, role: str) -> None:
    with storage.edit(pid) as p:
        f = _file(p, fid)
        if not f:
            raise KeyError(fid)
        f.role, f.role_set, f.status, f.detail = role, True, "queued", "reading again with the new role"  # type: ignore[assignment]
        _log(p, "file_role_set", file=fid, role=role)


def delete_file(pid: str, fid: str) -> None:
    with storage.edit(pid) as p:
        f = _file(p, fid)
        if not f:
            raise KeyError(fid)
        _remove_file_content(p, fid)
        storage.remove_file(pid, f"uploads/{f.stored_as}")
        p.sources = [x for x in p.sources if x.id != fid]
        A.say(p, f"I removed {f.filename} and everything that came only from it.", kind="text")
        _log(p, "file_removed", file=fid)
    sources.remove_source(pid, fid)


# ================================================================ conversation flow

def mentor_on() -> bool:
    return agent_backend() is not None


def _files_event(p: Project) -> bool:
    """With the mentor on: once every upload is read, tell it so it can respond (first impressions, feedback,
    what is missing). Posted as a hidden event message the chat worker hands to the mentor."""
    if processing(p):
        return False
    announced = {fid for m in p.messages if m.data.get("event") for fid in m.data.get("files", [])}
    fresh = [f for f in p.sources if f.status in ("ready", "error") and f.id not in announced]
    if not fresh:
        return False
    desc = "; ".join(f"{f.id} {f.filename} ({'could not be read' if f.status == 'error' else f'{f.claims} facts'})" for f in fresh)
    first = not announced
    ask = ("Give the researcher your first read as a mentor: what the work is about, what looks strong, what a reviewer "
           "would question, and the one or two most important things the paper still needs from them."
           if first else "Tell the researcher briefly what the new material adds and what is still missing.")
    p.messages.append(A.ChatMessage(id=A.next_message_id(p), role="user", text=f"Finished reading {desc}. {ask}",
                                    data={"event": True, "files": [f.id for f in fresh]}))
    return True


def _after_file(p: Project) -> bool:
    """Next step after a file finished or failed. Returns True when the mentor has something to answer."""
    if mentor_on():
        return _files_event(p)
    _advance(p)
    return False


def _advance(p: Project) -> None:
    """Decide the next conversational step after something changed. No model calls."""
    if mentor_on():
        return                              # the mentor decides what to ask
    if processing(p) or A.open_question(p) or p.stage == "writing":
        return
    if not p.ledger.claims:
        return
    need = A.next_question(p)
    if need:
        A.ask(p, need)
        p.stage = "questions"
        return
    if p.draft is None:
        p.outline = planner.plan(p.ledger, p.settings.target_pages)
        if p.stage != "ready":            # present the plan once; later additions just update it
            A.plan_message(p)
            p.stage = "ready"


def _swap_absorbed(p: Project, absorbed: list[tuple[str, str]]) -> set[str]:
    """A new claim absorbed an older, vaguer one: it takes the older one's place in the outline.
    Returns the new claim IDs that were placed that way."""
    placed: set[str] = set()
    for old, new in absorbed:
        _log(p, "claim_absorbed", old=old, new=new)
        if p.outline and planner.replace_claim(p.outline, old, new):
            placed.add(new)
    return placed


def _chat_passage(p: Project, message_id: str, text: str) -> Passage:
    n = int(message_id[1:]) if message_id[1:].isdigit() else 0
    return Passage(id=f"chat:{message_id}", source_id="chat", location=f"message {n}", text=text)


def _apply_claims(pid: str, found: list[X.Checked], passage: Passage, question_id: str | None) -> tuple[list, bool]:
    """Add claims from a chat message; returns (added claims, whether a draft needs rewriting)."""
    sources.add(pid, [passage])
    with storage.edit(pid) as p:
        absorbed: list[tuple[str, str]] = []
        added, merged = X.merge_into(p.ledger, found, origin="chat", nli=True, absorbed=absorbed)
        placed = _swap_absorbed(p, absorbed)
        if p.outline:
            for c in added:
                if c.id not in placed:
                    planner.place_claim(p.outline, p.ledger, c)
            planner.refresh_abstract(p.outline, p.ledger)
        q = next((x for x in p.questions if x.id == question_id), None)
        if q:
            q.status = "answered" if (added or merged) else "open"
        if added:
            where = sorted({planner.section_for(c, p.ledger) for c in added})
            A.say(p, f"Got it. I added {len(added)} claim{'s' if len(added) != 1 else ''} for the "
                     f"{' and '.join(where)}: " + "; ".join(f"{c.id} {c.text}" for c in added[:3]) +
                  ("…" if len(added) > 3 else "."), data={"added": [c.id for c in added]})
        elif merged:
            A.say(p, "That's already in your material, so I linked your message to it.")
        else:
            A.say(p, "I couldn't find a statement to add in that. Tell me facts about the work in plain sentences, "
                     "upload a file, or say \"skip\".", options=["Skip this"] if q else [])
        _log(p, "chat_claims", added=[c.id for c in added], merged=merged)
        _advance(p)
        rewrite = bool(added) and p.draft is not None
    return added, rewrite


MENTOR_BYPASS = {"Write the paper", "Continue writing", "Write with what you have", "Export to Word", "Show what needs review"}


def handle_chat(pid: str, message_id: str, backend_name: str | None, use_nli: bool, job: jobs.Job) -> None:
    snap = storage.load(pid)
    msg = next((m for m in snap.messages if m.id == message_id), None)
    if not msg:
        return
    text = msg.text.strip()
    intent = A.classify(text)
    q = A.open_question(snap)
    job.message = "Thinking"

    # quick replies that name a file's role
    role_m = re.match(r"^(.+?) is (my own work|a paper to cite)$", text, re.I)
    if role_m:
        f = next((x for x in snap.sources if x.filename.lower() == role_m.group(1).strip().lower()), None)
        if f:
            role = "own" if "own" in role_m.group(2).lower() else "reference"
            set_role(pid, f.id, role)
            with storage.edit(pid) as p:
                A.say(p, f"Understood. I'm reading {f.filename} again as " + ("your own work." if role == "own" else "a paper to cite."))
            start_ingest(pid, f.id, backend_name, use_nli)
            return
    retry_m = re.match(r"^retry (.+)$", text, re.I)
    if retry_m:
        f = next((x for x in snap.sources if x.filename.lower() == retry_m.group(1).strip().lower()), None)
        if f:
            with storage.edit(pid) as p:
                ff = _file(p, f.id)
                ff.status, ff.detail = "queued", "trying again"
            start_ingest(pid, f.id, backend_name, use_nli)
            return

    # the mentor runs the conversation; only the fixed quick replies below skip it (instant, no model call)
    if msg.data.get("event") and not mentor_on():
        return
    if mentor_on() and (msg.data.get("event") or text not in MENTOR_BYPASS):
        from . import agent
        agent.respond(pid, message_id, use_nli, job)
        return

    if A.WRITE_ANYWAY_RE.match(text) or intent.kind == "write":
        with storage.edit(pid) as p:
            for x in p.questions:
                if x.status == "open":
                    x.status = "skipped"
            if not p.ledger.claims:
                A.say(p, "There's nothing to write from yet. Upload your material or tell me about the work first.")
                return
            if p.outline is None or (p.draft is None and not intent.section):
                p.outline = planner.plan(p.ledger, p.settings.target_pages)   # fresh plan with everything so far
            if intent.paragraph and p.outline:
                for s in p.outline.sections:
                    for para in s.paragraphs:
                        if para.id == intent.paragraph:
                            para.stale = True
            if _needs_writing(p, intent.section) is None:
                A.say(p, "The paper is up to date with everything you've given me. Ask me to change a section, add "
                         "material, or export it.", options=["Export to Word"])
                return
            if jobs.running(pid, "write"):
                A.say(p, "I'm already writing; I'll include that as part of this run.")
            else:
                A.say(p, f"Writing {'the ' + intent.section if intent.section else 'the paper'} now. You can keep chatting "
                         "while I write.")
        start_write(pid, backend_name, use_nli, section=intent.section)
        return

    if intent.kind == "review":
        with storage.edit(pid) as p:
            from .review import needs_review
            items = [s for dp in (p.draft.paragraphs if p.draft else []) for s in dp.sentences if needs_review(s)]
            if not items:
                A.say(p, "Nothing needs review: every sentence passed verification or you accepted it.")
            else:
                lines = [f"{s.id}: \"{s.text[:110]}\" ({next(f.detail for f in s.flags if f.severity != 'info')})" for s in items[:6]]
                A.say(p, f"{len(items)} sentence{'s' if len(items) != 1 else ''} need{'s' if len(items) == 1 else ''} your review. "
                         "Click one in the paper to see its sources and fix it, or tell me how to change it:\n\n" + "\n".join(lines),
                      data={"review": [s.id for s in items]})
        return

    if intent.kind == "will_upload":
        with storage.edit(pid) as p:
            A.say(p, "Sure. Drop the file in whenever you're ready; I'll read it and carry on from there.")
        return

    if q and A.SKIP_RE.match(text):
        with storage.edit(pid) as p:
            qq = next(x for x in p.questions if x.id == q.id)
            qq.status = "skipped"
            A.say(p, "OK, I'll leave that out of the paper.")
            _advance(p)
        return

    if intent.kind == "pages":
        pages = max(1, min(40, int(intent.value)))
        with storage.edit(pid) as p:
            p.settings.target_pages = pages
            if p.outline:
                planner.apply_targets(p.outline, p.ledger.template, pages)
            A.say(p, f"Target set to {pages} pages. " + ("Send your material whenever you're ready." if not p.ledger.claims
                                                         else "I'll size the sections to that."))
            _log(p, "target_pages", pages=pages)
            if p.ledger.claims and not processing(p):
                _advance(p)
        return

    if intent.kind == "title":
        with storage.edit(pid) as p:
            p.ledger.title = intent.value[:300]
            A.say(p, f"Title set: \"{p.ledger.title}\".")
        return

    if intent.kind == "authors":
        names = [n.strip() for n in re.split(r",|;|\band\b", intent.value) if n.strip()]
        with storage.edit(pid) as p:
            p.ledger.authors = names
            A.say(p, "Authors set: " + ", ".join(names) + ".")
        return

    if intent.kind == "template":
        with storage.edit(pid) as p:
            p.ledger.template = intent.value  # type: ignore[assignment]
            if p.outline:
                planner.apply_targets(p.outline, p.ledger.template, p.settings.target_pages)
            label = {"ieee": "IEEE conference (two columns)", "acm": "ACM (acmart)", "article": "single-column article"}[intent.value]
            A.say(p, f"Switched to the {label} template.")
        return

    if intent.kind == "export":
        with storage.edit(pid) as p:
            links = [{"label": "Word (.docx)", "href": f"/api/projects/{pid}/export/docx"},
                     {"label": "LaTeX (.tex)", "href": f"/api/projects/{pid}/export/latex"},
                     {"label": "Submission bundle (.zip)", "href": f"/api/projects/{pid}/export/zip"},
                     {"label": "AI-use disclosure (.md)", "href": f"/api/projects/{pid}/export/disclosure"}]
            if p.draft is None:
                A.say(p, "There's no draft yet. Say \"write the paper\" first.")
            else:
                flagged = provenance.stats(p)["flagged"]
                note = f" {flagged} sentence{'s' if flagged != 1 else ''} still need your review first." if flagged else ""
                A.say(p, "Here are your downloads." + note + " For a PDF, open the Export tab and choose Print.", data={"links": links})
        return

    if intent.kind == "status":
        with storage.edit(pid) as p:
            A.say(p, A.status_text(p))
        return

    if intent.kind == "remove":
        ids = intent.value.split(",")
        with storage.edit(pid) as p:
            known = p.ledger.claim_map()
            gone = [i for i in ids if i in known]
            p.ledger.claims = [c for c in p.ledger.claims if c.id not in gone]
            if p.outline:
                for cid in gone:
                    planner.remove_claim(p.outline, cid)
            A.say(p, (f"Removed {', '.join(gone)}." if gone else "I couldn't find those claim numbers.")
                  + (" I'll rewrite the paragraphs that used them." if gone and p.draft else ""))
            rewrite = bool(gone) and p.draft is not None
        if rewrite:
            start_write(pid, backend_name, use_nli)
        return

    if intent.kind == "restyle":
        instruction, factor = A.style_instruction(text)
        with storage.edit(pid) as p:
            if not p.outline or not p.draft:
                A.say(p, "There's nothing written yet to change. Say \"write the paper\" first.")
                return
            touched = 0
            for s in p.outline.sections:
                if intent.section and s.name != intent.section:
                    continue
                if factor != 1.0:
                    s.target_words = int((s.target_words or 150 * len(s.paragraphs)) * factor)
                for para in s.paragraphs:
                    if intent.paragraph and para.id != intent.paragraph:
                        continue
                    para.style = instruction
                    para.stale = True
                    touched += 1
            where = intent.paragraph or (f"the {intent.section}" if intent.section else "the whole paper")
            how = f": \"{instruction}\". Wording only; the content stays exactly what your material says." if instruction else \
                ". Same claims, a fresh attempt at the wording."
            A.say(p, f"Rewriting {where} ({touched} paragraph{'s' if touched != 1 else ''}){how}" if touched
                  else "I couldn't find that part of the paper.")
            _log(p, "restyle", target=where, instruction=instruction)
        if touched:
            start_write(pid, backend_name, use_nli)
        return

    if intent.kind == "ask":
        passages = sources.load(pid)
        hits = sources.search(passages, text, k=5)
        files = {f.id: f.filename for f in snap.sources}
        excerpts = [(_label(files, ps), ps.text) for ps, _ in hits]
        answer, found = "", False
        if excerpts:
            try:
                with llm_lock(pid):
                    raw = get_backend(backend_name).generate_json(A.QA_SYSTEM, A.qa_prompt(text, excerpts), A.QA_SCHEMA)
                answer, found = str(raw.get("answer", "")).strip(), bool(raw.get("found"))
            except BackendError as exc:
                answer = f"I couldn't reach the local model to answer that ({exc})."
        with storage.edit(pid) as p:
            if found and answer:
                A.say(p, answer, data={"sources": [{"label": lab, "text": t[:300]} for lab, t in excerpts[:3]]})
            else:
                A.say(p, answer or "I couldn't find that in your material. If it belongs in the paper, tell me and I'll add it.")
        return

    # information about the work (possibly the answer to an open question)
    default_type = None
    if q:
        need = A.NEED_BY_ID.get(q.need)
        default_type = need.types[0] if need else None
    passage = _chat_passage(snap, message_id, text)
    try:
        with llm_lock(pid):
            found = X.claims_from_answer(get_backend(backend_name, role="read"), text, passage, default_type, use_nli=use_nli)
    except BackendError:
        found = X.claims_from_answer(None, text, passage, default_type, use_nli=False)
    _added, rewrite = _apply_claims(pid, found, passage, q.id if q else None)
    if rewrite:
        start_write(pid, backend_name, use_nli)


def _label(files: dict[str, str], ps: Passage) -> str:
    if ps.source_id == "chat":
        return f"your {ps.location}"
    if ps.source_id == "suggestion":
        return ps.location
    return f"{files.get(ps.source_id, ps.source_id)}, {ps.location}"


# ================================================================ writing

def evidence_for(p: Project, passages: dict[str, Passage], claim_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
    files = {f.id: f.filename for f in p.sources}
    known = p.ledger.claim_map()
    out: dict[str, list[tuple[str, str]]] = {}
    for cid in claim_ids:
        c = known.get(cid)
        if not c:
            continue
        items = []
        for sid in c.sources[:2]:
            ps = passages.get(sid)
            if ps and not ps.id.startswith(tuple(f"{f}.t" for f in files)):   # tables reach the writer as tables
                # only the part of the passage about this claim: the writer and the verifier both see this
                items.append((_label(files, ps), focused_excerpt(c.text, ps.text)))
        if items:
            out[cid] = items
    return out


def _needs_writing(p: Project, section_filter: str | None) -> tuple[str, object] | None:
    written = {dp.id for dp in p.draft.paragraphs} if p.draft else set()
    for s in p.outline.sections if p.outline else []:
        if section_filter and s.name != section_filter:
            continue
        for para in s.paragraphs:
            if para.id not in written or para.stale:
                return s.name, para
    return None


def _earlier(p: Project, para_id: str) -> list[DraftParagraph]:
    """Written paragraphs that come before para_id in the paper."""
    if not p.draft or not p.outline:
        return []
    order = [pp.id for s in p.outline.sections for pp in s.paragraphs]
    if para_id not in order:
        return []
    before = set(order[: order.index(para_id)])
    return [dp for dp in p.draft.paragraphs if dp.id in before]


def _order_draft(p: Project) -> None:
    if not p.draft or not p.outline:
        return
    order = {para.id: i for i, para in enumerate(pp for s in p.outline.sections for pp in s.paragraphs)}
    p.draft.paragraphs = sorted((dp for dp in p.draft.paragraphs if dp.id in order), key=lambda dp: order[dp.id])


def start_write(pid: str, backend_name: str | None, use_nli: bool, section: str | None = None) -> jobs.Job | None:
    try:
        return jobs.start(pid, "write", lambda job: write_job(pid, backend_name, use_nli, section, job))
    except jobs.JobConflict:
        return None                    # the running writer picks up the new work: it rereads the outline


def write_job(pid: str, backend_name: str | None, use_nli: bool, section_filter: str | None, job: jobs.Job) -> None:
    backend = get_backend(backend_name)
    with storage.edit(pid) as p:
        if p.outline is None:
            p.outline = planner.plan(p.ledger, p.settings.target_pages)
        elif not any(s.target_words for s in p.outline.sections):
            planner.apply_targets(p.outline, p.ledger.template, p.settings.target_pages)
        written_ids = {d.id for d in p.draft.paragraphs} if p.draft else set()
        total = sum(1 for s in p.outline.sections if not section_filter or s.name == section_filter
                    for para in s.paragraphs if para.stale or para.id not in written_ids)
        if total == 0:
            return                          # nothing new to write
        if p.draft is None:
            p.draft = Draft(backend=backend.name, model=backend.model)
        _order_draft(p)
        p.stage = "writing"
        progress = A.say(p, "Writing…", kind="progress", data={"done": 0, "total": total, "section": ""})
        progress_id = progress.id
        _log(p, "write_started", backend=backend.name, model=backend.model, paragraphs=total)
    job.total = total
    done = 0
    failures = 0
    try:
        while True:
            p = storage.load(pid)
            nxt = _needs_writing(p, section_filter)
            if nxt is None:
                break
            section_name, para = nxt
            section = next(s for s in p.outline.sections if s.name == section_name)
            snapshot = list(para.claim_ids)
            passages = sources.by_id(pid)
            evidence = evidence_for(p, passages, para.claim_ids)
            prev = ""
            if p.draft:
                ids = [pp.id for pp in section.paragraphs]
                before = ids[: ids.index(para.id)]
                prev_dp = next((d for d in reversed(p.draft.paragraphs) if d.id in before), None)
                prev = writer.paragraph_text(prev_dp) if prev_dp else ""
            job.message = f"Writing {section_name}"

            def waiting(seconds: float, name: str = section_name) -> None:
                job.message = f"Writing {name}: waiting {round(seconds)} s for the model's rate limit"

            try:
                with llm_lock(pid), report_waits(waiting):
                    written = writer.write_paragraph(backend, p.ledger, section_name, para, prev, use_nli,
                                                     evidence=evidence, target_words=planner.paragraph_target(section),
                                                     earlier=_earlier(p, para.id))
            except BackendError as exc:
                if is_fatal(exc):
                    raise
                failures += 1
                written = DraftParagraph(id=para.id, section=section_name, claim_ids=list(para.claim_ids), attempts=3,
                                         issues=[Flag(kind="not_written", severity="error",
                                                      detail=f"the model could not write this paragraph ({exc}); say \"rewrite {para.id}\" to try again")])
            with storage.edit(pid) as p2:
                live = next((pp for s in p2.outline.sections for pp in s.paragraphs if pp.id == para.id), None)
                if live is None:
                    continue                                   # removed while it was being written
                verifier.flag_repeats(written, _earlier(p2, para.id))
                if live.claim_ids == snapshot:
                    live.stale = False
                p2.draft = p2.draft or Draft(backend=backend.name, model=backend.model)
                p2.draft.paragraphs = [d for d in p2.draft.paragraphs if d.id != para.id] + [written]
                _order_draft(p2)
                p2.draft.updated_at = now_iso()
                done += 1
                msg = next((m for m in p2.messages if m.id == progress_id), None)
                if msg:
                    msg.data = {"done": done, "total": max(total, done), "section": section_name}
                    msg.text = f"Writing {section_name}…"
            job.progress = done
    except BackendError as exc:
        with storage.edit(pid) as p3:
            p3.stage = "revising" if p3.draft and p3.draft.paragraphs else "ready"
            how = ("Start PaperSmith with PaperSmithAI.bat so the local model is running, then say \"continue writing\"."
                   if is_local() else "Everything written so far is kept; say \"continue writing\" when the model is available again.")
            A.say(p3, f"I had to stop writing: {exc}. {how}", kind="error", options=["Continue writing"])
        raise
    with storage.edit(pid) as p4:
        p4.consistency = consistency.check_all(p4.ledger, p4.draft, use_nli=False)
        p4.stage = "revising"
        msg = next((m for m in p4.messages if m.id == progress_id), None)
        if msg:
            msg.text, msg.data = "Writing finished.", {"done": done, "total": max(total, done), "section": ""}
        st = provenance.stats(p4)
        words = sum(len(s.text.split()) for dp in p4.draft.paragraphs for s in dp.sentences) if p4.draft else 0
        pages = round(words / planner.words_per_page(p4.ledger.template), 1)
        text = f"Done: about {pages:g} page{'s' if pages != 1 else ''} ({st['sentences']} sentences)."
        if st["flagged"]:
            text += f" {st['flagged']} sentence{'s' if st['flagged'] != 1 else ''} need{'s' if st['flagged'] == 1 else ''} your review."
        if failures:
            text += f" {failures} paragraph{'s' if failures != 1 else ''} couldn't be written; say \"continue writing\" to try again."
        opts = ["Show what needs review"] if st["flagged"] else []
        opts += ["Review it like a reviewer", "Export to Word"] if mentor_on() else ["Make it longer", "Export to Word"]
        A.say(p4, text, kind="done", options=opts, data={"pages": pages, "sentences": st["sentences"], "flagged": st["flagged"]})
        _log(p4, "write_finished", paragraphs=done, sentences=st["sentences"], flagged=st["flagged"])


def start_ingest(pid: str, fid: str, backend_name: str | None, use_nli: bool) -> jobs.Job | None:
    try:
        return jobs.start(pid, "ingest", lambda job: ingest_file(pid, fid, backend_name, use_nli, job), key=fid)
    except jobs.JobConflict:
        return None


_chat_guards: dict[str, threading.Lock] = {}
_chat_alive: set[str] = set()


def _chat_guard(pid: str) -> threading.Lock:
    with _locks_guard:
        return _chat_guards.setdefault(pid, threading.Lock())


def post_user_message(pid: str, text: str, author: str = "", author_name: str = "") -> str:
    with storage.edit(pid) as p:
        mid = A.next_message_id(p)
        from .models import ChatMessage
        p.messages.append(ChatMessage(id=mid, role="user", text=text.strip()[:8000], author=author, author_name=author_name))
        if p.stage == "welcome":
            p.stage = "collecting"
    return mid


def start_chat(pid: str, backend_name: str | None, use_nli: bool) -> None:
    """One worker per project handles unhandled user messages strictly in order."""
    with _chat_guard(pid):
        if pid in _chat_alive:
            return                         # the running worker will pick the message up
        _chat_alive.add(pid)
    # ordering is guaranteed by _chat_alive, so each worker gets its own job key (a finishing worker's
    # job may still read "running" for a moment and must not block the next one)
    jobs.start(pid, "chat", lambda job: _chat_worker(pid, backend_name, use_nli, job), key=f"w{time.time_ns()}")


def _chat_worker(pid: str, backend_name: str | None, use_nli: bool, job: jobs.Job) -> None:
    while True:
        with _chat_guard(pid):
            p = storage.load(pid)
            nxt = next((m for m in p.messages if m.role == "user" and not m.data.get("handled")), None)
            if nxt is None:
                _chat_alive.discard(pid)
                return
        with storage.edit(pid) as p2:
            live = next(m for m in p2.messages if m.id == nxt.id)
            live.data = {**live.data, "handled": True}
        try:
            handle_chat(pid, nxt.id, backend_name, use_nli, job)
        except Exception as exc:  # noqa: BLE001 - one bad message must not stop the conversation
            log.exception("chat message %s failed", nxt.id)
            with storage.edit(pid) as p3:
                A.say(p3, f"Something went wrong while handling that ({type(exc).__name__}). Please try again.", kind="error")


def recover(pid: str) -> None:
    """After a restart the jobs are gone: mark interrupted work so the author can resume it."""
    if jobs.running(pid):
        return
    snap = storage.load(pid)
    stuck = [f.id for f in snap.sources if f.status in ("queued", "reading", "extracting")]
    if not stuck and snap.stage != "writing":
        return
    with storage.edit(pid) as p:
        for f in p.sources:
            if f.id in stuck:
                f.status, f.detail = "error", "interrupted when PaperSmith stopped; choose Retry"
                A.say(p, f"Reading {f.filename} was interrupted when PaperSmith stopped.", kind="error",
                      options=["Retry " + f.filename], data={"file": f.id})
        if p.stage == "writing":
            p.stage = "revising" if p.draft and p.draft.paragraphs else "ready"
            A.say(p, "Writing was interrupted when PaperSmith stopped. Say \"continue writing\" to finish the remaining paragraphs.",
                  options=["Continue writing"])
