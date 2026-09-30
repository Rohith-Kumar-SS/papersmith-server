"""PaperSmith's mentor.

A language model runs the conversation: it sees the state of the paper, the researcher's material and the
chat, talks like a supervisor, gives feedback, asks for what is missing and decides what to do next. It never
writes the paper itself. Facts it hears reach the paper only as claims that pass the same check as file
extraction (numbers, terms, wording, NLI) against the message or passage they came from; the writer and the
verifier then work from those claims exactly as before.

A turn is one model call in the common case: reply, facts and actions come back as one JSON object. When the
mentor needs to see more first (the draft, every fact, a search), it asks for that and is called once more.
"""

from __future__ import annotations

import logging
import re

from . import assistant as A
from . import extract as X
from . import jobs, planner, sources, storage
from . import textutils as tu
from . import workspace as W
from .config import settings
from .llm import BackendError, agent_backend, get_backend, report_waits
from .models import ChatMessage, ClaimType, Passage, Project
from .review import needs_review

log = logging.getLogger(__name__)

TYPES = [t.value for t in ClaimType]
ACTIONS = ["write", "rewrite", "read_paper", "list_facts", "search", "set_title", "set_authors", "set_template",
           "set_pages", "remove_fact", "set_file_role", "export"]
CONTEXT_ACTIONS = ("read_paper", "list_facts", "search")

PROMPT_CHARS = 15000        # with the system prompt this keeps a turn inside an 8k tokens-per-minute free tier
MAX_FACTS_PER_TURN = 12

SYSTEM = """You are PaperSmith, a research mentor and paper-writing partner working with one researcher on one paper. Be the experienced supervisor they wish they had: warm, direct, practical, and honest about weaknesses. Help them see what their work shows, what reviewers will ask, and what the paper still needs, and write the paper with them.

How the paper gets written
- The paper is built only from facts about the work that come from the researcher: their uploaded material (passage IDs such as D1.p4) or what they tell you in this chat (message IDs such as M12). A writer turns facts into prose and a verifier checks every sentence against them, so nothing that is not recorded as a fact can reach the paper.
- Record facts generously. Whenever the researcher's latest message says anything about their work, even casually (what they are doing and why, the setting, data, method, numbers, results, interpretation, limitations, plans, background they want stated), put each distinct point in "facts": one short, complete, self-contained sentence ("We ...", not a fragment like "so cities can ..."), with source = that message ID. Facts are checked word for word against that message, so reuse its words and copy numbers and units exactly; never add details, numbers or conclusions they did not state. Skip only questions, requests, greetings and points already listed as facts. Never ask for something they have just told you: record it and build on it.
- Plans for the paper itself (where to submit, length, title, authors, deadlines) are not facts about the research: use set_template, set_pages, set_title or set_authors instead.
- When the researcher says a fact is wrong or does not belong in the paper, remove it with remove_fact (and record the corrected version if they give one), and confirm it in your reply.
- Before saying something is missing, check FACTS SO FAR and RELEVANT MATERIAL: praise and question what is actually there.
  Example. M4 researcher: "I'm calibrating cheap PM2.5 sensors so cities can monitor pollution without expensive reference stations. We logged 24 sensors for 30 days." facts: [{"text": "We are calibrating cheap PM2.5 sensors", "type": "objective", "source": "M4"}, {"text": "Cheap sensors let cities monitor pollution without expensive reference stations", "type": "background", "source": "M4"}, {"text": "We logged 24 sensors for 30 days", "type": "dataset", "source": "M4"}]
- To record something from RELEVANT MATERIAL that is not yet a fact, use its passage ID as source.
- You may suggest ideas: framing, analyses, baselines, limitations, future work, structure. Mark them clearly as suggestions. Record a suggestion as a fact only when the researcher's latest message accepts it; then source = "S:" + the ID of your message that made it, and keep that message's wording.
- Never say you wrote, changed or added something unless this reply includes the action or fact that does it.

How to talk
- Be a mentor, not a form: react to what they said with substance (what is strong, what a reviewer will question, what is missing or unclear), then ask for what you need next. At most two questions per reply, most important first, each with a few words on why it matters.
- Keep replies short, usually under 150 words, in plain paragraphs or a short "- " list; use **bold** sparingly. Go longer only for reviews and explanations they ask for.
- Answer questions about their work from RELEVANT MATERIAL and the facts; if the answer is not there, say so. You may explain general research and writing practice, clearly as general advice, never as a fact about their work.
- Offer to write once the facts cover the objective, method, data and main results. Be honest about length: the state says how many pages the facts support; tell them what extra material would make the paper longer. Never pad.
- To review the draft, first request read_paper, then critique like a reviewer: summary, strengths, major issues, minor issues, and what the researcher could supply to fix each.
- Events (not messages from the researcher) tell you something happened, such as files finishing reading; respond to the researcher about it.

Actions: list of {kind, target, value}; use "" when a field is not needed.
- write: write every paragraph not written yet. target: "" or one section (Abstract, Introduction, Related Work, Methods, Results, Discussion, Conclusion) or a paragraph ID.
- rewrite: rewrite written text. target: a section, a paragraph ID or "all". value: a wording instruction ("more formal", "shorter", "longer", "active voice") or "" for a fresh attempt. Wording only; content always comes from the facts.
- read_paper (target: a section or ""), list_facts (target: a fact type or ""), search (value: a query): see more before answering. Use them alone with an empty reply; you will be called again with the result.
- set_title (value), set_authors (value: names separated by commas), set_template (value: ieee, acm or article), set_pages (value: a number).
- remove_fact (target: fact ID). set_file_role (target: file ID, value: own, reference or data).
- export: show the download links (Word, LaTeX, zip).

Fact types: background (known in the field), gap (what prior work lacks), objective, contribution, method, dataset, result, comparison (a result against a baseline), interpretation (the researcher's reading of a result), limitation, future_work.

"options": up to 3 short replies the researcher might tap next, such as "Write the paper" or "Review it like a reviewer"; [] if none fit. They appear as buttons: never repeat them in the reply text."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "facts", "actions", "options"],
    "properties": {
        "reply": {"type": "string"},
        "facts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "type", "source"],
            "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": TYPES}, "source": {"type": "string"}}}},
        "actions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["kind", "target", "value"],
            "properties": {"kind": {"type": "string", "enum": ACTIONS}, "target": {"type": "string"}, "value": {"type": "string"}}}},
        "options": {"type": "array", "items": {"type": "string"}},
    },
}

NEGATIVE = re.compile(r"^\s*(no|nope|nah|don'?t|do not|not really|skip|never ?mind|rather not|leave it)\b", re.I)
# a researcher asking to change or drop something they said; the mentor may delete a fact only then
CHANGE = re.compile(r"\b(remove|delete|drop|take (?:it|that|this|them) out|get rid|wrong|incorrect|not (?:true|right|correct)|"
                    r"shouldn'?t|should not|doesn'?t belong|does not belong|replace|instead|actually|correction|mistake|"
                    r"not in the paper|leave (?:it|that|this) out)\b", re.I)
# statements about the paper itself, not the research: they go to settings (title, venue), never into the paper
META = re.compile(r"\b(title|authors?|co-?authors?|publish(?:ed|ing)?|submi(?:t|ssion)|conference|journal|venue|page limit|"
                  r"pages?|deadline|this paper|the paper|manuscript|upload|slides)\b", re.I)
COMMAND = re.compile(r"^\s*(please\s+)?(write|rewrite|review|export|continue|show|remove|delete|make|set|use|change|"
                     r"yes|no|ok|okay|thanks|thank you|sure)\b", re.I)
TEMPLATES = {"ieee": "IEEE conference", "acm": "ACM", "article": "single-column article"}
ROLES = {"own": "own work", "reference": "paper to cite", "data": "data"}


# ================================================================ context

def _files_line(p: Project) -> str:
    parts = []
    for f in p.sources:
        if f.status == "ready":
            cited = f", cited as [{f.reference_key}]" if f.reference_key else ""
            parts.append(f"{f.id} {f.filename} ({ROLES.get(f.role, f.role)}, {f.claims} facts{cited})")
        elif f.status == "error":
            parts.append(f"{f.id} {f.filename} (could not be read: {f.detail[:80]})")
        else:
            parts.append(f"{f.id} {f.filename} (still reading)")
    return "; ".join(parts) or "none yet"


def state_text(p: Project) -> str:
    counts = A.count_by_type(p)
    wpp = planner.words_per_page(p.ledger.template)
    lines = ["PAPER STATE",
             f"Title: {p.ledger.title or '(not set)'} | Authors: {', '.join(p.ledger.authors) or '(not set)'} | "
             f"Template: {TEMPLATES.get(p.ledger.template, p.ledger.template)} | Target length: {p.settings.target_pages} pages",
             f"Files: {_files_line(p)}",
             f"Facts: {len(p.ledger.claims)}" + (" (" + ", ".join(f"{t} {n}" for t, n in counts.items()) + ")" if counts else "")]
    thin = [n.id.replace("_", " ") for n in A.missing_needs(p)]
    if thin:
        lines.append("Not covered by any fact yet: " + ", ".join(thin))
    if p.ledger.claims:
        outline = p.outline or planner.plan(p.ledger, p.settings.target_pages)
        supported = round(planner.supported_words(p.ledger, outline) / wpp, 1)
        lines.append(f"The facts support about {supported:g} of the {p.settings.target_pages} target pages.")
    if p.draft and p.draft.paragraphs:
        total = sum(len(s.paragraphs) for s in p.outline.sections) if p.outline else len(p.draft.paragraphs)
        words = sum(len(s.text.split()) for dp in p.draft.paragraphs for s in dp.sentences)
        review = sum(1 for dp in p.draft.paragraphs for s in dp.sentences if needs_review(s))
        lines.append(f"Draft: {len(p.draft.paragraphs)} of {total} paragraphs written, about {words / wpp:.1f} pages; "
                     f"{review} sentence{'s' if review != 1 else ''} flagged for the researcher's review.")
    else:
        lines.append("Draft: not written yet.")
    if jobs.running(p.id, "write"):
        lines.append("The writer is working on the paper right now.")
    if p.members:
        lines.append("Team: " + ", ".join(f"{m.name} ({'co-author' if m.role == 'author' else 'reviewer'})" for m in p.members)
                     + ". Several researchers may write in this chat; messages show who wrote them.")
    open_reviews = [r for r in p.reviews if not r.resolved]
    if open_reviews:
        lines.append("OPEN REVIEW COMMENTS (from the paper's reviewers; help the researcher address them): " + " | ".join(
            f"{r.id} on {r.target} by {r.author_name or 'a reviewer'}: {r.text[:220]}" for r in open_reviews[:6]))
    hint = community_hint(p)
    if hint:
        lines.append(hint)
    if p.ledger.claims:
        lines.append(facts_digest(p))
    last = next((m for m in reversed(p.messages) if m.role == "assistant" and m.kind == "text"), None)
    if last and last.data.get("rejected"):
        lines.append("Facts you proposed last turn that failed the check against the researcher's words (ask them to "
                     "confirm or restate): " + "; ".join(f"\"{r['text'][:100]}\" ({r['reason']})" for r in last.data["rejected"][:4]))
    return "\n".join(lines)


FACTS_CHARS = 5500


def facts_digest(p: Project, chars: int = FACTS_CHARS) -> str:
    """Every fact, grouped by type, so the mentor knows what the paper already has. A ledger too long for
    the budget keeps the newest facts of each type and says how many more there are (list_facts shows all)."""
    by_type: dict[str, list] = {}
    for c in p.ledger.claims:
        by_type.setdefault(c.type.value, []).append(c)

    def render(per_type: int, width: int) -> str:
        lines = ["FACTS SO FAR"]
        for t in TYPES:
            items = by_type.get(t, [])
            if not items:
                continue
            shown = items[-per_type:]
            more = f" (+{len(items) - len(shown)} more; list_facts shows them)" if len(items) > len(shown) else ""
            lines.append(f"{t}{more}: " + " | ".join(
                f"{c.id}{' [prior work ' + ', '.join(c.refs) + ']' if c.refs else ''} {c.text[:width]}" for c in shown))
        return "\n".join(lines)

    most = max((len(v) for v in by_type.values()), default=0)
    for per_type, width in [(most, 140), (most, 90), *[(n, 90) for n in range(most - 1, 1, -1)], (2, 70)]:
        text = render(per_type, width)
        if len(text) <= chars:
            return text
    return render(2, 70)[:chars]


def community_hint(p: Project) -> str:
    """People in the researcher's college whose published profile fits this paper, for the mentor to suggest.
    Only their public profile (name, role, topics) is used, and only when the researcher is in the community."""
    if not p.owner:
        return ""
    try:
        from .community import interests, matching, related
        from .community.store import store as community_store

        s = community_store()
        me = s.get_person(p.owner)
        if not me or not me.get("institution"):
            return ""
        lines = []
        doc = s.get_doc("paper_topics", p.id)
        if doc:                              # people at the college on this paper's own topics
            found = related.find(doc, limit=3)
            lines += [f"{x['person']['name']} ({x['person']['role_label'] or 'researcher'}; works on {', '.join(x['shared'])})"
                      for x in found["people"]]
            lines += [f"open project “{o['title']}” by {o['owner_name']} (on {', '.join(o['shared'])})" for o in found["openings"][:1]]
            lines += [f"ongoing college project “{r['title']}” led by {r['lead'] or 'a colleague'}" for r in found["projects"][:1]]
        if not lines:
            ranked = matching.rank(me, s.members(me["institution"]), s.network(p.owner), limit=3,
                                   private=interests.private_interests(me))
            lines = [f"{x['person']['name']} ({x['person']['role_label'] or 'researcher'}; {x['reasons'][0] if x['reasons'] else ''})"
                     for x in (ranked["mentors"][:2] + ranked["collaborators"][:2])[:3]]
    except Exception:  # noqa: BLE001 - the community is a bonus; the mentor works without it
        log.exception("community hint failed")
        return ""
    if not lines:
        return ""
    return ("COMMUNITY (people and projects in the researcher's college on the same work; suggest one when it fits, e.g. "
            "for a reviewer, a gap or a collaborator, and tell the researcher to find them under Community): " + " | ".join(lines[:4]))


def _who(m: ChatMessage, team: bool = False) -> str:
    if m.role == "user":
        if m.data.get("event"):
            return "event"
        return f"researcher {m.author_name}" if team and m.author_name else "researcher"
    return "you" if m.data.get("mentor") else "note"


def history_text(p: Project, reply_to: str, budget: int) -> str:
    msgs = [m for m in p.messages if m.kind != "progress"]
    idx = next((i for i, m in enumerate(msgs) if m.id == reply_to), len(msgs) - 1)
    lines: list[str] = []
    used = 0
    for m in reversed(msgs[: idx + 1]):
        limit = 4000 if m.id == reply_to else 600
        text = re.sub(r"\s+", " ", m.text).strip()
        line = f"{m.id} {_who(m, bool(p.members))}: {text[:limit]}{'…' if len(text) > limit else ''}"
        if lines and (used + len(line) > budget or len(lines) >= 16):
            break
        lines.append(line)
        used += len(line)
    return "\n".join(reversed(lines))


def _label(p: Project, ps: Passage) -> str:
    files = {f.id: f.filename for f in p.sources}
    return W._label(files, ps)


def material_text(p: Project, query: str, k: int = 5, chars: int = 450) -> str:
    if not query.strip():
        return ""
    hits = sources.search(sources.load(p.id), query, k=k)
    return "\n".join(f"[{ps.id}] ({_label(p, ps)}) {re.sub(chr(10) + '+', ' / ', ps.text)[:chars]}" for ps, _ in hits)


def paper_text(p: Project, section: str = "", chars: int = 9000) -> str:
    if not p.draft or not p.draft.paragraphs:
        return "PAPER TEXT\n(nothing written yet)"
    lines, current = ["PAPER TEXT"], ""
    for dp in p.draft.paragraphs:
        if section and dp.section.lower() != section.lower():
            continue
        if dp.section != current:
            current = dp.section
            lines.append(f"\n{current}")
        flagged = sum(1 for s in dp.sentences if needs_review(s))
        note = f" [{flagged} flagged]" if flagged else ""
        lines.append(f"{dp.id}{note}: " + " ".join(s.text for s in dp.sentences))
    return "\n".join(lines)[:chars]


def facts_text(p: Project, ctype: str = "", chars: int = 7000) -> str:
    items = [c for c in p.ledger.claims if not ctype or c.type.value == ctype]
    return ("ALL FACTS" + (f" ({ctype})" if ctype else "") + "\n" +
            "\n".join(f"{c.id} ({c.type.value}) {c.text}" for c in items))[:chars]


def _context_block(p: Project, action: dict) -> str:
    kind, target, value = action["kind"], action["target"].strip(), action["value"].strip()
    if kind == "read_paper":
        return paper_text(p, A.SECTION_NAME.get(target.lower(), target) if target else "")
    if kind == "list_facts":
        return facts_text(p, target if target in TYPES else "")
    return f"SEARCH RESULTS for \"{value or target}\"\n" + (material_text(p, value or target, k=8, chars=500) or "(nothing found)")


def build_prompt(p: Project, msg: ChatMessage, extra: list[str], budget: int = PROMPT_CHARS,
                 recorded: list | None = None) -> str:
    state = state_text(p)
    event = bool(msg.data.get("event"))
    material = "" if event or extra else material_text(p, msg.text)
    blocks = "\n\n".join(extra)
    if extra:
        blocks += "\n\nYou asked to see the above. Now reply to the researcher; do not ask for more context."
    room = budget - len(state) - len(material) - len(blocks) - 200
    history = history_text(p, msg.id, max(1200, room))
    parts = [state]
    if material:
        parts.append("RELEVANT MATERIAL (from the researcher's files and messages)\n" + material)
    if blocks:
        parts.append(blocks)
    parts.append("CONVERSATION (oldest first)\n" + history)
    checklist = ("Before you answer: (1) never ask for anything already in FACTS SO FAR; question or build on it instead; "
                 "(2) every request in the message to change the paper or its facts needs its action (remove_fact, "
                 "set_title, write, rewrite, ...).")
    if event:
        parts.append(f"Reply to: {msg.id} (an event, not a message from the researcher). {checklist}")
    else:
        got = (f"Already recorded from {msg.id} and now in the paper: " +
               "; ".join(f"{c.id} ({c.type.value}) {c.text[:120]}" for c in recorded) + ". Do not repeat them in facts."
               if recorded else f"Nothing from {msg.id} has been recorded yet.")
        parts.append(f"Reply to: {msg.id}. {checklist} (3) {got} Put any other statement {msg.id} makes about the work "
                     f"in facts (source {msg.id}); facts already in FACTS SO FAR never go in facts again. (4) Describe "
                     "only changes that really happened this turn.")
    return "\n\n".join(parts)


# ================================================================ facts

def _entails(premise: str, hypothesis: str) -> float:
    from .nli import get_scorer

    scorer = get_scorer()
    if not scorer or not scorer.load():
        return 0.0
    return scorer.score([(premise[:1800], hypothesis)])[0][0]


def check_facts(p: Project, msg: ChatMessage, items: list[dict], use_nli: bool):
    """Verify the facts the mentor proposes against the words they cite.
    Returns (accepted [(Checked, origin, refs)], rejected [{text, reason}], new passages to store)."""
    by_msg = {m.id: m for m in p.messages}
    files = {f.id: f for f in p.sources}
    known = [X._signature(c.text) for c in p.ledger.claims]
    stored: dict[str, Passage] | None = None
    accepted, rejected, new_passages = [], [], {}
    for item in items[:MAX_FACTS_PER_TURN * 2]:
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip().rstrip(".")
        text = re.sub(r"^\[prior work [^\]]*\]\s*", "", text)
        src = str(item.get("source", "")).strip().replace("chat:", "")
        if not text or re.fullmatch(r"C\d+", src, re.I):
            continue                        # restating a fact the paper already has
        sig = X._signature(text)
        if sig and any(k and len(sig & k) / len(sig | k) >= 0.75 for k in known):
            continue
        if len(accepted) + len(rejected) >= MAX_FACTS_PER_TURN:
            break
        try:
            ctype = ClaimType(item.get("type"))
        except ValueError:
            ctype = X.guess_type(text)
        evidence, passage, origin, refs, reason = None, None, "chat", [], ""
        sugg = re.fullmatch(r"S:?\s*(M\d+)", src, re.I)
        if sugg:
            sm = by_msg.get(sugg.group(1).upper())
            if not sm or sm.role != "assistant":
                reason = "the suggestion it cites does not exist"
            elif msg.role != "user" or msg.data.get("event") or NEGATIVE.match(msg.text):
                reason = "you have not accepted this suggestion"
            else:
                evidence, origin = sm.text, "suggested"
                passage = Passage(id=f"chat:{sm.id}", source_id="suggestion", text=sm.text,
                                  location=f"PaperSmith's suggestion in message {sm.id[1:]}, accepted in message {msg.id[1:]}")
        elif re.fullmatch(r"M\d+", src, re.I):
            um = by_msg.get(src.upper())
            if not um or um.role != "user" or um.data.get("event"):
                reason = "its source is not one of your messages"
            elif META.search(text):
                continue                    # about the paper itself (title, venue): a setting, not a fact
            else:
                evidence, passage = um.text, W._chat_passage(p, um.id, um.text)
        else:
            stored = stored if stored is not None else sources.by_id(p.id)
            ps = stored.get(src)
            if not ps:
                reason = "its source could not be found"
            else:
                evidence = ps.text
                f = files.get(ps.source_id)
                origin = "file" if f else "chat"
                if f and f.role == "reference":
                    refs = [f.reference_key] if f.reference_key else []
                    if ctype not in (ClaimType.background, ClaimType.gap):
                        ctype = ClaimType.background
        if evidence is not None:
            reason = X.check_claim(text, evidence)
            # paraphrase of the researcher's own words: accept when the entailment model agrees
            if reason.startswith("only") and use_nli and _entails(evidence, text) >= 0.6:
                reason = ""
        if reason:
            rejected.append({"text": text, "reason": reason})
            continue
        sid = passage.id if passage else src
        accepted.append((X.Checked(text=text, type=ctype, passages=[sid], ok=True), origin, refs))
        if passage:
            new_passages[passage.id] = passage
    return accepted, rejected, list(new_passages.values())


def add_facts(p: Project, accepted, protect: set[str] | None = None) -> list:
    """protect: fact IDs the message names ("C12 is wrong, it was 20"): new facts are never merged into them."""
    added_all = []
    groups: dict[tuple[str, tuple[str, ...]], list[X.Checked]] = {}
    for checked, origin, refs in accepted:
        groups.setdefault((origin, tuple(refs)), []).append(checked)
    held = [c for c in p.ledger.claims if protect and c.id in protect]
    p.ledger.claims = [c for c in p.ledger.claims if not (protect and c.id in protect)]
    absorbed: list[tuple[str, str]] = []
    try:
        for (origin, refs), items in groups.items():
            added, _merged = X.merge_into(p.ledger, items, origin=origin, refs=list(refs) or None, nli=True, absorbed=absorbed)
            added_all += added
    finally:
        if held:
            p.ledger.claims = sorted(held + p.ledger.claims, key=lambda c: int(c.id[1:]) if c.id[1:].isdigit() else 0)
    placed = W._swap_absorbed(p, absorbed)
    if p.outline and added_all:
        for c in added_all:
            if c.id not in placed:
                planner.place_claim(p.outline, p.ledger, c)
        planner.refresh_abstract(p.outline, p.ledger)
    return added_all


# ================================================================ actions

def _target(target: str) -> tuple[str | None, str | None]:
    t = target.strip()
    if re.fullmatch(r"P\d+", t, re.I):
        return None, t.upper()
    if t.lower() in ("", "all", "paper", "the paper", "everything"):
        return None, None
    return A.SECTION_NAME.get(t.lower(), t.title()), None


def _act(p: Project, action: dict, follow: dict, ctx: dict | None = None) -> str:
    """Apply one action to the project inside the caller's edit. Returns a note for the researcher when the
    action could not be done; follow-up jobs go into `follow`. ctx: allow_remove (the researcher asked for a
    change) and passage (this message's receipt: a fact it just restated is not deleted)."""
    ctx = ctx or {}
    kind, target, value = action["kind"], action["target"].strip(), action["value"].strip()
    if kind == "write":
        if not p.ledger.claims:
            return "There are no facts to write from yet."
        section, para = _target(target)
        if p.outline is None or (p.draft is None and not section):
            p.outline = planner.plan(p.ledger, p.settings.target_pages)
        if para:
            for s in p.outline.sections:
                for pp in s.paragraphs:
                    if pp.id == para:
                        pp.stale = True
        if W._needs_writing(p, section) is None:
            return "The paper is already up to date with every fact."
        follow["write"] = True
        follow.setdefault("sections", set()).add(section)
    elif kind == "rewrite":
        if not p.outline or not p.draft:
            return "Nothing is written yet, so there is nothing to rewrite."
        section, para = _target(target)
        instruction, factor = A.style_instruction(value) if value else ("", 1.0)
        touched = 0
        for s in p.outline.sections:
            if section and s.name != section:
                continue
            if factor != 1.0 and not para:
                s.target_words = int((s.target_words or 150 * len(s.paragraphs)) * factor)
            for pp in s.paragraphs:
                if para and pp.id != para:
                    continue
                pp.style, pp.stale = instruction, True
                touched += 1
        if not touched:
            return f"I couldn't find {target or 'that part'} in the paper."
        follow["write"] = True
        follow.setdefault("sections", set()).add(None)
        W._log(p, "restyle", target=target or "all", instruction=instruction)
    elif kind == "set_title" and value:
        p.ledger.title = value[:300]
    elif kind == "set_authors" and value:
        p.ledger.authors = [n.strip() for n in re.split(r",|;|\band\b", value) if n.strip()]
    elif kind == "set_template":
        tpl = value.lower()
        if tpl not in TEMPLATES:
            return "The template must be ieee, acm or article."
        p.ledger.template = tpl  # type: ignore[assignment]
        if p.outline:
            planner.apply_targets(p.outline, p.ledger.template, p.settings.target_pages)
    elif kind == "set_pages":
        m = re.search(r"\d+", value or target)
        if not m:
            return ""
        p.settings.target_pages = max(1, min(40, int(m.group())))
        if p.outline:
            planner.apply_targets(p.outline, p.ledger.template, p.settings.target_pages)
    elif kind == "remove_fact":
        known = p.ledger.claim_map()
        asked = [c for c in re.findall(r"C\d+", f"{target} {value}".upper()) if c in known]
        if not asked:
            return "I couldn't find that fact."
        if not ctx.get("allow_remove"):
            return f"I kept {', '.join(asked)} in the paper; tell me if it should come out."
        gone = [c for c in asked if ctx.get("passage") not in known[c].sources]
        if not gone:
            return ""
        p.ledger.claims = [c for c in p.ledger.claims if c.id not in gone]
        if p.outline:
            for cid in gone:
                planner.remove_claim(p.outline, cid)
        if p.draft:
            follow["write"] = True
            follow.setdefault("sections", set()).add(None)
    elif kind == "set_file_role":
        f = W._file(p, target.upper())
        if not f or value not in ROLES:
            return "I couldn't change that file's role."
        f.role, f.role_set, f.status, f.detail = value, True, "queued", "reading again with the new role"  # type: ignore[assignment]
        follow.setdefault("ingest", []).append(f.id)
    elif kind == "export":
        if p.draft is None:
            return "There's no draft to download yet."
        follow["links"] = [{"label": "Word (.docx)", "href": f"/api/projects/{p.id}/export/docx"},
                           {"label": "LaTeX (.tex)", "href": f"/api/projects/{p.id}/export/latex"},
                           {"label": "Submission bundle (.zip)", "href": f"/api/projects/{p.id}/export/zip"}]
    return ""


_OPTIONS_IN_REPLY = re.compile(r"\n\s*(?:\*\*)?(?:options|quick replies|suggested replies)(?:\*\*)?\s*:.*$", re.I | re.S)


def _clean(raw: dict) -> dict:
    facts = [f for f in raw.get("facts", []) if isinstance(f, dict)]
    actions = [{"kind": str(a.get("kind", "")), "target": str(a.get("target", "")), "value": str(a.get("value", ""))}
               for a in raw.get("actions", []) if isinstance(a, dict) and a.get("kind") in ACTIONS]
    options = [str(o).strip()[:60] for o in raw.get("options", []) if str(o).strip()][:3]
    reply = _OPTIONS_IN_REPLY.sub("", str(raw.get("reply", ""))).strip()   # quick replies are shown as buttons
    return {"reply": reply, "facts": facts, "actions": actions, "options": options}


# ================================================================ a turn

def _worth_reading(msg: ChatMessage) -> bool:
    """Does the message state anything about the work (not just a question, a request or a quick reply)?"""
    if msg.role != "user" or msg.data.get("event") or msg.attachments:
        return False
    text = msg.text.strip()
    if len(text.split()) < 5:
        return False
    return any(not s.rstrip().endswith("?") and not COMMAND.match(s) for s in tu.split_sentences(text) if s.strip())


def _record_message(pid: str, msg: ChatMessage, use_nli: bool) -> list:
    """Extract the facts a chat message states with the same extractor and checks as uploaded files."""
    snap = storage.load(pid)
    passage = W._chat_passage(snap, msg.id, msg.text)
    try:
        res = X.extract(get_backend(role="read"), "the researcher's chat message", "own", [passage], use_nli=use_nli)
    except BackendError as exc:
        log.info("reading %s failed (%s); the mentor records its facts instead", msg.id, exc)
        return []
    found = [c for c in res.accepted if not META.search(c.text)]
    if not found:
        return []
    sources.add(pid, [passage])
    named = set(re.findall(r"\bC\d+\b", msg.text.upper()))
    with storage.edit(pid) as p:
        return add_facts(p, [(c, "chat", []) for c in found], protect=named)


def _available(backend, chars: int):
    """The mentor's model, or the fallback model when the mentor's per-minute allowance is used up."""
    wait = getattr(backend, "expected_wait", None)
    estimate = int(chars / 3.3) + 1500
    if wait is None or not settings.groq_fallback_model or wait(estimate) <= 8:
        return backend
    from .llm import OpenAICompatBackend

    alt = OpenAICompatBackend(model=settings.groq_fallback_model, effort=settings.groq_agent_effort)
    if alt.model != backend.model and alt.expected_wait(estimate) <= 1:
        log.info("%s is rate-limited; %s answers this turn", backend.model, alt.model)
        return alt
    return backend


def respond(pid: str, message_id: str, use_nli: bool, job: jobs.Job) -> None:
    backend = agent_backend()
    snap = storage.load(pid)
    msg = next((m for m in snap.messages if m.id == message_id), None)
    if not msg or backend is None:
        return
    if msg.text.strip().lower() in ("try again", "retry"):
        failed = next((m for m in reversed(snap.messages) if m.role == "user" and m.data.get("failed")), None)
        if failed:
            msg = failed

    def waiting(seconds: float) -> None:
        job.message = f"Waiting {round(seconds)} s for the model's rate limit"

    # 1) the careful extractor records what the message says about the work, before the mentor answers, so the
    #    mentor's reply can only describe what really went into the paper
    job.message = "Thinking"
    recorded: list = []
    if _worth_reading(msg):
        with report_waits(waiting):
            recorded = _record_message(pid, msg, use_nli)
        if recorded:
            snap = storage.load(pid)
            msg = next(m for m in snap.messages if m.id == msg.id)

    # 2) the mentor's turn; when its model is rate-limited, the fallback model answers instead of waiting
    backend = _available(backend, len(SYSTEM) + PROMPT_CHARS)
    turn, first, extra, budget, retried = None, None, [], PROMPT_CHARS, False
    with report_waits(waiting):
        for _ in range(4):
            try:
                raw = backend.generate_json(SYSTEM, build_prompt(snap, msg, extra, budget, recorded), SCHEMA,
                                            max_tokens=4000, temperature=0.5)
            except BackendError as exc:
                if "too large" in str(exc) and budget > 5000:
                    budget //= 2
                    continue
                if not retried and ("JSON" in str(exc) or "length cap" in str(exc)):
                    retried = True              # an occasional malformed or cut-off answer: ask once more
                    continue
                if first:
                    turn = first
                    break
                _failed(pid, msg.id, exc)
                return
            turn = _clean(raw)
            wants = [a for a in turn["actions"] if a["kind"] in CONTEXT_ACTIONS]
            if wants and not extra:
                first = turn if turn["reply"] else None
                extra = [_context_block(snap, a) for a in wants[:2]]
                job.message = "Reading the paper" if any(a["kind"] == "read_paper" for a in wants) else "Looking through your material"
                continue
            break
    if turn is None:
        return

    accepted, rejected, new_passages = check_facts(snap, msg, turn["facts"], use_nli)
    if new_passages:
        sources.add(pid, new_passages)
    follow: dict = {}
    named = set(re.findall(r"\bC\d+\b", msg.text.upper()))
    ctx = {"allow_remove": msg.role == "user" and not msg.data.get("event") and bool(named or CHANGE.search(msg.text)),
           "passage": f"chat:{msg.id}"}
    with storage.edit(pid) as p:
        # removals first, so a correction ("C12 is wrong, it was 20") replaces instead of merging into the old fact
        actions = sorted((a for a in turn["actions"] if a["kind"] not in CONTEXT_ACTIONS), key=lambda a: a["kind"] != "remove_fact")
        notes = []
        added = []
        for a in actions:
            if a["kind"] != "remove_fact" and not added and accepted:
                added = add_facts(p, accepted, protect=named)
                accepted = []
            note = _act(p, a, follow, ctx)
            if note:
                notes.append(note)
        if accepted:
            added = add_facts(p, accepted, protect=named)
        live = p.ledger.claim_map()
        added = [c for c in recorded if c.id in live] + added
        data: dict = {"mentor": True}
        if added:
            data["facts"] = [{"id": c.id, "type": c.type.value, "text": c.text} for c in added]
        if rejected:
            data["rejected"] = rejected
        if notes:
            data["notes"] = notes
        if follow.get("links"):
            data["links"] = follow["links"]
        reply = turn["reply"] or ("Done." if turn["actions"] or added else "Could you say a little more?")
        A.say(p, reply, options=turn["options"], data=data)
        for m in p.messages:
            if m.id == msg.id and m.data.get("failed"):
                m.data = {k: v for k, v in m.data.items() if k != "failed"}
        if p.stage == "welcome":
            p.stage = "collecting"
        W._log(p, "mentor_turn", reply_to=msg.id, facts=[c.id for c in added], rejected=len(rejected),
               actions=[a["kind"] for a in turn["actions"]])
        write_after = bool(follow.get("write")) or (bool(added) and p.draft is not None)
        sections = follow.get("sections", {None})
    for fid in follow.get("ingest", []):
        W.start_ingest(pid, fid, None, use_nli)
    if write_after:
        section = next(iter(sections)) if len(sections) == 1 else None
        W.start_write(pid, None, use_nli, section=section)


def _failed(pid: str, message_id: str, exc: BackendError) -> None:
    text = str(exc)
    if "limit" in text:
        reply = (f"I've hit the model's usage limit ({text}). Your message is saved: tap Try again once it frees up, "
                 "or switch to the local model in Settings.")
    elif "API key" in text:
        reply = f"I can't use the model: {text}. Add the key to the .env file in the PaperSmith folder and restart PaperSmith."
    else:
        reply = f"I couldn't get an answer from the model ({text}). Tap Try again in a moment."
    with storage.edit(pid) as p:
        for m in p.messages:
            if m.id == message_id:
                m.data = {**m.data, "failed": True}
        A.say(p, reply, kind="error", options=["Try again"])
