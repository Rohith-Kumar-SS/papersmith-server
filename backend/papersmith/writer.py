"""Constrained writer: one paragraph at a time, verified, rewritten on failure."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from . import textutils as tu, verifier
from .config import settings
from .llm import BackendError, LLMBackend, _ungarble, is_fatal
from .models import Draft, DraftParagraph, Flag, Ledger, Outline, OutlineParagraph, Sentence
from .prompts import WRITER_SYSTEM, writer_schema, writer_user_prompt

log = logging.getLogger(__name__)


def paragraph_text(p: DraftParagraph) -> str:
    return " ".join(s.text for s in p.sentences)


# Models sometimes echo their bookkeeping into the prose: "(claim_ids=[C5])", "[C3, C4]", "(C2)".
_ID = r"""['"]?C\d+['"]?"""
_ID_ECHO_RE = re.compile(
    rf"\s*[(\[]\s*(?:claim_ids?|claims?|ref_keys?)?\s*[=:]?\s*\[?\s*{_ID}(?:\s*,\s*{_ID})*\s*\]?\s*[)\]]", re.I)


_ASIDE_RE = re.compile(r"\s*[(\[]([^()\[\]]*)(?:[)\]]|$)")
_ID_PART_RE = re.compile(
    rf"^\s*(?:see|cf\.?|from)?\s*(?:(?:claim_ids?|claims?|ref_keys?)\s*[=:]?\s*)?\[?\s*(?:{_ID}(?:\s+(?:and|&)\s+{_ID})*)?\s*\]?\s*[.]?\s*$", re.I)


def _drop_id_asides(text: str) -> str:
    """'(Table 1, Claims C27, C28)' -> '(Table 1)'; '(see C27)' -> ''. Only parts of an aside that are
    nothing but claim IDs go; any other wording in it stays as written."""
    def fix(m: re.Match) -> str:
        inner = m.group(1)
        if not re.search(r"\bC\d+\b|\bclaim(?:s|_ids?)?\b", inner, re.I):
            return m.group(0)
        parts = re.split(r"[,;]|\band\b", inner)
        kept = [x.strip() for x in parts if not _ID_PART_RE.match(x)]
        if len(kept) == len(parts):
            return m.group(0)
        return f" ({', '.join(kept)})" if kept else ""
    return _ASIDE_RE.sub(fix, text)


def clean_text(text: str, ref_keys: list[str] | None = None) -> str:
    """Formatting only: drop echoed claim IDs / citation keys, fix spacing, capitalise, end with punctuation."""
    text = _ungarble(text)
    text = _ID_ECHO_RE.sub("", text)
    text = _drop_id_asides(text)
    if ref_keys:
        key = "|".join(re.escape(k) for k in sorted(ref_keys, key=len, reverse=True))
        item = rf"(?:C\d+|{key})"
        text = re.sub(rf"\s*[(\[]\s*{item}(?:\s*[,;:]\s*{item})*\s*[)\]]", "", text)
        text = re.sub(rf"\b(?:{key})\b", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "approx.", "No.", "Ref.", "cf.")


def _sentences_of(text: str) -> list[str]:
    parts: list[str] = []
    for piece in tu.split_sentences(text):
        if parts and parts[-1].endswith(_ABBREVIATIONS):
            parts[-1] = f"{parts[-1]} {piece}"
        else:
            parts.append(piece)
    return parts


def split_item(text: str, claim_ids: list[str], ref_keys: list[str], ledger: Ledger | None) -> list[tuple[str, list[str], list[str]]]:
    """Split one model 'sentence' that holds several sentences, giving each piece the claims it states.

    A piece that states none of its item's claims is left untraced so the verifier rejects it, which
    is exactly what an appended "This highlights the need for ..." deserves.
    """
    parts = _sentences_of(text)
    known = ledger.claim_map() if ledger else {}
    ids = [c for c in claim_ids if c in known]
    if len(parts) <= 1 or not ids:
        return [(text, claim_ids, ref_keys)]
    cover = [{c: verifier.claim_coverage(known[c].text, part) for c in ids} for part in parts]
    assigned = [[c for c in ids if cv[c] >= 0.3] for cv in cover]
    for i, cv in enumerate(cover):
        best = max(cv, key=cv.get)
        if not assigned[i] and cv[best] >= 0.15:
            assigned[i] = [best]
    for c in ids:  # every claim the model tagged stays attached somewhere
        if not any(c in a for a in assigned):
            assigned[max(range(len(parts)), key=lambda i: cover[i][c])].append(c)
    out = []
    for part, part_ids in zip(parts, assigned):
        part_ids = [c for c in ids if c in part_ids]
        refs = [r for r in ref_keys if any(r in known[c].refs for c in part_ids)]
        out.append((part, part_ids, refs))
    return out


def tighten_receipts(sentences: list[Sentence], ledger: Ledger | None) -> None:
    """Drop a claim tag from a sentence that barely mentions it when another sentence clearly states it.

    Models over-tag ("C3, C4, C6, C7" on a sentence stating only C3 and C4). A receipt should show what its
    sentence says; the claim stays receipted where it is actually stated.
    """
    if not ledger:
        return
    known = ledger.claim_map()
    cover = {(s.id, c): verifier.claim_coverage(known[c].text, s.text) for s in sentences for c in s.claim_ids if c in known}
    for s in sentences:
        if len(s.claim_ids) < 2:
            continue
        keep = []
        for c in s.claim_ids:
            weak_here = cover.get((s.id, c), 1.0) < 0.2
            stated_elsewhere = any(o is not s and c in o.claim_ids and cover.get((o.id, c), 0) >= 0.4 for o in sentences)
            if not (weak_here and stated_elsewhere):
                keep.append(c)
        if keep and keep != s.claim_ids:
            dropped = set(s.claim_ids) - set(keep)
            s.claim_ids = keep
            s.ref_keys = [r for r in s.ref_keys
                          if any(r in known[c].refs for c in keep if c in known) or not any(r in known[c].refs for c in dropped if c in known)]


def _to_paragraph(raw: dict, outline_para: OutlineParagraph, section: str, attempt: int,
                  ref_keys: list[str] | None = None, ledger: Ledger | None = None) -> DraftParagraph:
    sentences = []
    for item in raw.get("sentences", []):
        text = clean_text(str(item.get("text", "")), ref_keys)
        if not text:
            continue
        item_ids = list(dict.fromkeys(str(c) for c in item.get("claim_ids", [])))
        # a sentence that carries claims is not a signpost, whatever the model says; one that carries none is
        # filler at best ("This section describes ...") and a disguised untraced repeat at worst: drop it
        if not item_ids:
            continue
        signpost = False
        item_refs = [str(r) for r in item.get("ref_keys", []) if str(r).strip()]
        pieces = [(text, item_ids, item_refs)] if signpost else split_item(text, item_ids, item_refs, ledger)
        for piece, ids, refs in pieces:
            sentences.append(Sentence(
                id=f"{outline_para.id}.S{len(sentences) + 1}",
                text=clean_text(piece, ref_keys),
                claim_ids=ids,
                ref_keys=refs,
                signpost=signpost,
            ))
    tighten_receipts(sentences, ledger)
    return DraftParagraph(id=outline_para.id, section=section, claim_ids=list(outline_para.claim_ids),
                          sentences=sentences, attempts=attempt)


def claim_sentence(text: str) -> str:
    """A claim stated plainly in the author's own wording. Table-row claims ("Model = Uncorrected;
    R2 = 0.61") become "For model Uncorrected, R2 was 0.61"."""
    parts = [piece.split("=", 1) for piece in text.split(";")]
    if len(parts) >= 2 and all(len(kv) == 2 and kv[0].strip() and kv[1].strip() for kv in parts):
        (k0, v0), *rest = [(k.strip(), v.strip()) for k, v in parts]
        values = [f"{k} was {v}" for k, v in rest]
        joined = values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]
        label = k0 if k0[:2].isupper() else k0[:1].lower() + k0[1:]
        return clean_text(f"For {label} {v0}, {joined}")
    return clean_text(text)


FALLBACK_BELOW = 0.35      # a tagged claim with less of its content in the text than this counts as left out


def repair(para: DraftParagraph, ledger: Ledger, earlier: list[DraftParagraph], use_nli: bool,
           evidence: verifier.Evidence | None) -> DraftParagraph:
    """Last resort once the rewrite attempts are used up, so an author's claim is never lost: sentences
    that restate an earlier paragraph go (the reader has already seen that content), and every claim the
    model left out, or only tagged, is stated plainly in the claim's own wording."""
    known = ledger.claim_map()
    before = len(para.sentences)
    kept = [s for s in para.sentences if not any(f.kind == "repeats_earlier" for f in s.flags)]
    missing = []
    for cid in para.claim_ids:
        if cid not in known:
            continue
        expressing = [s for s in kept if cid in s.claim_ids]
        said = " ".join(s.text for s in expressing)
        # "left out" = not stated at all, or flagged as under-expressed (a paraphrase the entailment model
        # confirmed carries no flag) with almost none of its content in the text
        flagged = any(f.kind == "claim_underexpressed" and f"with {cid}," in f.detail for s in expressing for f in s.flags)
        if not said or (flagged and verifier.claim_coverage(known[cid].text, said) < FALLBACK_BELOW):
            missing.append(cid)
    if not missing and len(kept) == len(para.sentences):
        return para
    for s in kept:
        s.claim_ids = [c for c in s.claim_ids if c not in missing]
        s.ref_keys = [r for r in s.ref_keys if any(r in known[c].refs for c in s.claim_ids if c in known)]
    kept = [s for s in kept if s.claim_ids or s.signpost]      # left with no claim, it stated none of the author's
    added = []
    for cid in missing:
        added.append(Sentence(id="", text=claim_sentence(known[cid].text), claim_ids=[cid],
                              ref_keys=[r for r in known[cid].refs if r in ledger.ref_map()]))
    para.sentences = kept + added
    for i, s in enumerate(para.sentences, 1):
        s.id = f"{para.id}.S{i}"
    verifier.verify_paragraph(para, ledger, use_nli, evidence)
    verifier.flag_repeats(para, earlier)
    for s in added:
        s.flags.append(Flag(kind="claim_wording", severity="info",
                            detail=f"the model left {s.claim_ids[0]} out, so it is stated in your own words"))
    log.info("%s repaired: %d sentence(s) dropped, %s stated as written", para.id,
             before - len(kept), ", ".join(missing) or "no claims")
    return para


def write_paragraph(
    backend: LLMBackend,
    ledger: Ledger,
    section: str,
    outline_para: OutlineParagraph,
    previous_paragraph: str = "",
    use_nli: bool = True,
    evidence: verifier.Evidence | None = None,
    target_words: int = 0,
    earlier: list[DraftParagraph] | None = None,
) -> DraftParagraph:
    """earlier: paragraphs that come before this one in the paper; a sentence restating them is rejected."""
    claims_by_id = ledger.claim_map()
    claims = [claims_by_id[c] for c in outline_para.claim_ids if c in claims_by_id]
    ref_keys = sorted({r for c in claims for r in c.refs if r in ledger.ref_map()})
    schema = writer_schema([c.id for c in claims], ref_keys)
    earlier = earlier or []

    best: DraftParagraph | None = None
    feedback: list[str] | None = None
    last_error: BackendError | None = None
    for attempt in range(1, settings.max_rewrite_attempts + 2):
        prompt = writer_user_prompt(ledger, section, claims, previous_paragraph, feedback,
                                    evidence=evidence, style=outline_para.style, target_words=target_words)
        try:
            raw = backend.generate_json(WRITER_SYSTEM, prompt, schema)
        except BackendError as exc:
            if is_fatal(exc):
                raise                       # unreachable, bad key or quota used up: retrying will not help
            last_error = exc
            log.info("%s attempt %d failed: %s", outline_para.id, attempt, exc)
            feedback = [f"Your previous output was unusable ({exc}). Write one concise paragraph; list each claim ID once."]
            continue
        para = verifier.verify_paragraph(_to_paragraph(raw, outline_para, section, attempt, ref_keys, ledger),
                                         ledger, use_nli, evidence)
        verifier.flag_repeats(para, earlier)
        if best is None or verifier.paragraph_score(para) < verifier.paragraph_score(best):
            best = para
        if verifier.paragraph_score(best) == 0:
            break
        feedback = verifier.feedback_for(para)
        log.info("%s attempt %d rejected: %s", outline_para.id, attempt, "; ".join(feedback)[:500])
    if best is None:
        raise last_error or BackendError("the writer produced no usable output")
    best.attempts = attempt
    if verifier.paragraph_score(best) > 0:
        best = repair(best, ledger, earlier, use_nli, evidence)
    return best


def write_draft(
    backend: LLMBackend,
    ledger: Ledger,
    outline: Outline,
    on_paragraph: Callable[[DraftParagraph, int, int], None] | None = None,
    use_nli: bool = True,
) -> Draft:
    targets = [(s.name, p) for s in outline.sections for p in s.paragraphs]
    draft = Draft(backend=backend.name, model=backend.model)
    previous_by_section: dict[str, str] = {}
    for i, (section, para) in enumerate(targets, start=1):
        written = write_paragraph(backend, ledger, section, para, previous_by_section.get(section, ""), use_nli,
                                  earlier=list(draft.paragraphs))
        previous_by_section[section] = paragraph_text(written)
        draft.paragraphs.append(written)
        if on_paragraph:
            on_paragraph(written, i, len(targets))
    return draft
