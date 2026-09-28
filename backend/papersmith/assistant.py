"""Conversation logic for the chat workspace.

The assistant never makes up content. It reads the author's material, works out what each section of the
paper still needs, and asks. Wording of its own replies is templated (fast, predictable, and independent of
how capable the local model is); the model is only used for bounded tasks elsewhere: extracting claims,
answering questions from the author's passages, and writing paragraphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import planner
from .models import ChatMessage, ClaimType as T, OpenQuestion, Project, now_iso

# ---------------------------------------------------------------- what a paper needs

@dataclass(frozen=True)
class Need:
    id: str
    section: str
    types: tuple[T, ...]
    minimum: int
    question: str
    optional: bool = False
    options: tuple[str, ...] = ()


NEEDS: list[Need] = [
    Need("objective", "Introduction", (T.objective,), 1,
         "In one or two sentences: what does this work set out to do?"),
    Need("gap", "Introduction", (T.gap,), 1,
         "What is missing in existing work that this paper addresses?"),
    Need("method", "Methods", (T.method,), 2,
         "How did you do it? Describe the method or procedure step by step. Short notes are fine."),
    Need("dataset", "Methods", (T.dataset,), 1,
         "What data did you use: where does it come from, how much of it is there, and how was it collected?"),
    Need("result", "Results", (T.result, T.comparison), 2,
         "What are your main results? Exact numbers are best. You can also upload the results table as CSV or Excel.",
         options=("I'll upload a table",)),
    Need("background", "Introduction", (T.background,), 1,
         "What background should the introduction give: what is already known, and why does the problem matter? "
         "If you have papers to cite, upload their PDFs or a .bib file.",
         options=("I'll upload papers to cite",)),
    Need("contribution", "Introduction", (T.contribution,), 1,
         "What does this paper contribute? One line per contribution is enough."),
    Need("limitation", "Discussion", (T.limitation,), 1,
         "What are the limitations of this work?", optional=True),
    Need("interpretation", "Discussion", (T.interpretation,), 1,
         "How do you interpret your main results? I will only write the interpretation you give me.", optional=True),
    Need("future_work", "Conclusion", (T.future_work,), 1,
         "Is there future work you want to mention?", optional=True),
]
NEED_BY_ID = {n.id: n for n in NEEDS}

TYPE_LABEL = {
    "background": "background", "gap": "gaps in prior work", "objective": "objectives", "contribution": "contributions",
    "method": "methods", "dataset": "data", "result": "results", "comparison": "comparisons",
    "interpretation": "interpretations", "limitation": "limitations", "future_work": "future work",
}

SKIP_RE = re.compile(r"^\s*(skip( this| it)?|no|none|nothing|n/?a|not applicable|pass|later|i don'?t know|dont know|not sure|no idea)\s*[.!]?\s*$", re.I)
WRITE_ANYWAY_RE = re.compile(r"^\s*(write( it)? with what you have|just write( it)?|write anyway|go ahead( and write)?)\s*[.!]?\s*$", re.I)


def count_by_type(project: Project) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in project.ledger.claims:
        counts[c.type.value] = counts.get(c.type.value, 0) + 1
    return counts


def missing_needs(project: Project) -> list[Need]:
    counts = count_by_type(project)
    out = []
    for n in NEEDS:
        have = sum(counts.get(t.value, 0) for t in n.types)
        if have < n.minimum:
            out.append(n)
    return out


def next_question(project: Project) -> Need | None:
    """The most important need that is missing and has not been asked yet."""
    asked = {q.need for q in project.questions}
    for n in missing_needs(project):
        if n.id not in asked:
            return n
    return None


def open_question(project: Project) -> OpenQuestion | None:
    return next((q for q in reversed(project.questions) if q.status == "open"), None)


# ---------------------------------------------------------------- understanding the author's message

SECTIONS = ["abstract", "introduction", "related work", "methods", "method", "results", "result", "discussion", "conclusion"]
SECTION_NAME = {"abstract": "Abstract", "introduction": "Introduction", "intro": "Introduction", "related work": "Related Work",
                "methods": "Methods", "method": "Methods", "methodology": "Methods", "results": "Results", "result": "Results",
                "discussion": "Discussion", "conclusion": "Conclusion", "conclusions": "Conclusion"}
_SECTION_RE = re.compile(r"\b(abstract|introduction|intro|related work|methods?|methodology|results?|discussion|conclusions?)\b", re.I)
_PARA_RE = re.compile(r"\b(P\d+)(?:\.S\d+)?\b")


@dataclass
class Intent:
    kind: str                 # write | restyle | pages | title | authors | template | export | status | remove | ask | info
    section: str | None = None
    paragraph: str | None = None
    value: str = ""


def classify(text: str) -> Intent:
    t = text.strip()
    low = t.lower()
    section = None
    m = _SECTION_RE.search(low)
    if m:
        section = SECTION_NAME.get(m.group(1).lower())
    para = _PARA_RE.search(t)
    paragraph = para.group(1) if para else None

    if re.match(r"^(the\s+)?(working\s+)?title\s*(is|:|=|should be)\s*", low):
        return Intent("title", value=re.sub(r"^(the\s+)?(working\s+)?title\s*(is|:|=|should be)\s*", "", t, flags=re.I).strip(" \"'."))
    if re.match(r"^(the\s+)?authors?\s*(are|is|:|=)\s*", low):
        return Intent("authors", value=re.sub(r"^(the\s+)?authors?\s*(are|is|:|=)\s*", "", t, flags=re.I).strip(" ."))
    pages = re.search(r"\b(\d{1,2})\s*(?:-\s*page|pages?|pp)\b", low)
    if pages and len(low) < 80:
        return Intent("pages", value=pages.group(1))
    if re.search(r"\b(ieee|acm|single[- ]column|one[- ]column|article format)\b", low) and \
            re.search(r"\b(template|format|style|use|switch|make it|conference|journal)\b", low) and len(low) < 90:
        tpl = "ieee" if "ieee" in low else "acm" if "acm" in low else "article"
        return Intent("template", value=tpl)
    if re.search(r"\b(export|download|save (it )?as|give me the (file|paper)|\.docx|\.tex)\b", low) or \
            re.search(r"\b(word|docx|latex|pdf) (file|version|document)\b", low):
        return Intent("export")
    if re.search(r"\b(remove|delete|drop|forget)\b", low) and re.search(r"\bC\d+\b", t):
        return Intent("remove", value=",".join(re.findall(r"\bC\d+\b", t)))
    if re.match(r"^(status|progress|help|what'?s next|what now|what do you need( from me)?|where are we)\s*\??\s*$", low):
        return Intent("status")
    if re.search(r"\b(show|list|what)\b.*\b(needs? (my )?review|flagged|to review|problems)\b", low):
        return Intent("review")
    if re.match(r"^(i'?ll|i will|let me|going to) (upload|send|attach|add) ", low):
        return Intent("will_upload")
    restyle = re.search(r"\b(rewrite|rephrase|reword|shorten|shorter|expand|longer|more (formal|concise|detailed|academic|technical)|"
                        r"less (formal|technical|wordy)|simplify|simpler|tighten|polish|concise|clearer|passive voice|active voice)\b", low)
    if restyle and (section or paragraph or re.search(r"\b(it|this|the paper|whole paper|everything|all)\b", low)):
        return Intent("restyle", section=section, paragraph=paragraph, value=t)
    if re.match(r"^(please\s+)?(write|draft|generate|start( writing)?|continue( writing)?|finish( writing)?|go ahead|let'?s (write|start|go)|"
                r"update|redo|regenerate)\b", low) or WRITE_ANYWAY_RE.match(t):
        return Intent("write", section=section, paragraph=paragraph)
    if t.endswith("?"):
        return Intent("ask", value=t)
    return Intent("info", value=t)


def style_instruction(text: str) -> tuple[str, float]:
    """Turn a restyle request into a wording instruction and a length factor."""
    low = text.lower()
    factor = 1.0
    if re.search(r"\b(shorten|shorter|concise|tighten|trim|cut)\b", low):
        factor = 0.7
    elif re.search(r"\b(expand|longer|more detailed|elaborate)\b", low):
        factor = 1.4
    instruction = re.sub(r"\b(please|can you|could you)\b", "", text, flags=re.I).strip(" .?!")
    # a bare "rewrite the results" / "rewrite P7" asks for a fresh attempt, not a style change
    bare = re.sub(r"^(re-?write|redo|regenerate|rephrase|reword)\s+(the\s+)?", "", instruction, flags=re.I)
    bare = re.sub(r"\b(P\d+(\.S\d+)?|section|paragraph|abstract|introduction|intro|related work|methods?|methodology|"
                  r"results?|discussion|conclusions?|whole paper|the paper|paper|it|this|everything|all)\b", "", bare, flags=re.I)
    if factor == 1.0 and not re.sub(r"[\s:,.]+", "", bare):
        return "", factor
    instruction = re.sub(r"^(re-?write|rephrase|reword)\s+(P\d+|the \w+)\s*:\s*", "", instruction, flags=re.I) or instruction
    if factor > 1:
        instruction += " (use every supporting detail in the excerpts, but add nothing that is not in them)"
    return instruction, factor


# ---------------------------------------------------------------- messages

_seq = re.compile(r"^M(\d+)$")


def next_message_id(project: Project) -> str:
    nums = [int(m.id[1:]) for m in project.messages if m.id[1:].isdigit()]
    return f"M{(max(nums) if nums else 0) + 1}"


def say(project: Project, text: str, kind: str = "text", options: list[str] | None = None,
        data: dict | None = None, question_id: str | None = None) -> ChatMessage:
    msg = ChatMessage(id=next_message_id(project), role="assistant", text=text, kind=kind,  # type: ignore[arg-type]
                      options=list(options or []), data=dict(data or {}), question_id=question_id)
    project.messages.append(msg)
    return msg


def welcome(project: Project, mentor: bool = False) -> None:
    if mentor:
        say(project,
            "Hi, I'm PaperSmith. Tell me about your research, or share what you have: slides, drafts, notes, "
            "results or the papers you want to cite.\n\nWhat is your research about, and how long should the paper be?",
            options=["4 pages", "6 pages", "8 pages", "10 pages"], data={"mentor": True})
        return
    say(project,
        "Hi, I'm PaperSmith. I write your paper from your own material, and when something is missing I ask you "
        "instead of making it up. Every sentence I write links back to the slide, page or message it came from.\n\n"
        "Start by dropping in what you have: slides, a draft, thesis chapters, notes, results tables, or PDFs of papers "
        "you want to cite. Or just tell me about the work.\n\nHow long should the paper be?",
        options=["4 pages", "6 pages", "8 pages", "10 pages"])


def ask(project: Project, need: Need) -> OpenQuestion:
    qid = f"Q{len(project.questions) + 1}"
    missing = len(missing_needs(project))
    lead = "One thing I couldn't find in your material." if missing <= 1 else "Something your material doesn't cover yet."
    msg = say(project, f"{lead} {need.question}", kind="question",
              options=list(need.options) + ["Skip this", "Write with what you have"], question_id=qid,
              data={"need": need.id, "section": need.section})
    q = OpenQuestion(id=qid, need=need.id, section=need.section, text=need.question, asked_in=msg.id)
    project.questions.append(q)
    return q


def claims_summary(counts: dict[str, int]) -> str:
    parts = [f"{n} {TYPE_LABEL.get(t, t)}" for t, n in sorted(counts.items(), key=lambda x: -x[1]) if n]
    return ", ".join(parts)


def plan_message(project: Project) -> ChatMessage:
    outline = project.outline or planner.plan(project.ledger, project.settings.target_pages)
    wpp = planner.words_per_page(project.ledger.template)
    supported = planner.supported_words(project.ledger, outline)
    supported_pages = round(supported / wpp, 1)
    target = project.settings.target_pages
    sections = [{"name": s.name, "paragraphs": len(s.paragraphs),
                 "claims": sum(len(p.claim_ids) for p in s.paragraphs), "target_words": s.target_words}
                for s in outline.sections]
    thin = [n for n in missing_needs(project)]
    lines = [f"Here is the plan: {len(outline.sections)} sections from {len(project.ledger.claims)} claims."]
    if supported_pages + 0.5 < target:
        lines.append(f"Your material supports about {supported_pages:g} of the {target} pages you asked for. "
                     "I won't pad the paper with content you didn't give me.")
        if thin:
            lines.append("To make it longer, tell me more about: " + ", ".join(n.id.replace("_", " ") for n in thin[:4]) + ".")
        else:
            lines.append("To make it longer, add more detail on your methods and results, or upload more material.")
    else:
        lines.append(f"That is enough material for about {target} pages.")
    lines.append("Say \"write the paper\" when you're ready, or keep adding material.")
    return say(project, "\n\n".join(lines), kind="plan", options=["Write the paper"],
               data={"sections": sections, "supported_pages": supported_pages, "target_pages": target})


def status_text(project: Project) -> str:
    counts = count_by_type(project)
    files = [f for f in project.sources if f.status == "ready"]
    written = len(project.draft.paragraphs) if project.draft else 0
    total = sum(len(s.paragraphs) for s in project.outline.sections) if project.outline else 0
    lines = [f"So far: {len(files)} files read and {len(project.ledger.claims)} claims ({claims_summary(counts) or 'none yet'})."]
    if total:
        lines.append(f"{written} of {total} planned paragraphs are written.")
    need = next_question(project)
    q = open_question(project)
    if q:
        lines.append(f"I'm waiting for your answer to: {q.text}")
    elif need:
        lines.append(f"Still missing: {need.id.replace('_', ' ')}.")
    elif written < total or not total:
        lines.append("Say \"write the paper\" to start writing.")
    else:
        lines.append("The paper is written. Ask me to change any section, or export it from the Export tab.")
    return " ".join(lines)


# ---------------------------------------------------------------- grounded answers about the author's material

QA_SYSTEM = """You answer the author's question about their own research material. Use only the excerpts given. Quote numbers exactly. If the excerpts do not answer the question, say that you could not find it in their material. Keep the answer under 120 words. Return JSON."""

QA_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "found"],
    "properties": {"answer": {"type": "string"}, "found": {"type": "boolean"}},
}


def qa_prompt(question: str, excerpts: list[tuple[str, str]]) -> str:
    lines = [f"Question: {question}", "", "Excerpts from the author's material:"]
    lines += [f"({loc}) " + re.sub(r"\s+", " ", text[:600]) for loc, text in excerpts]
    return "\n".join(lines)


def touch(project: Project) -> None:
    project.updated_at = now_iso()
