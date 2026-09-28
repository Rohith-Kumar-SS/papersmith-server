"""Prompts and output schema for the constrained writer."""

from __future__ import annotations

import re

from . import textutils as tu
from .models import Claim, DataTable, Ledger

WRITER_SYSTEM = """You are PaperSmith, a scientific prose writer. You turn claims supplied by the authors into academic prose. You are a scribe, not a scientist: the ideas belong to the authors.

Rules:
1. Use only the information in the provided claims, their supporting excerpts from the author's material, tables and figures. Do not add facts, numbers, methods, datasets, names, comparisons, causes, explanations, implications, significance statements or future directions that these do not state. The excerpts may add detail to a claim; every such detail must appear in the excerpt of a claim you list for that sentence.
2. Do not evaluate the work. Words such as novel, important, promising, robust, significant or state-of-the-art may appear only if the claim itself uses them.
3. Do not correct or improve the science. If a claim looks questionable, express it as given.
4. Every sentence lists the IDs of the claims it expresses in claim_ids. Every claim you are given must be expressed at least once. You may merge claims into one sentence or split one claim across sentences, but list a claim ID on a sentence only if that sentence actually states the claim's content; do not drop any part of a claim. When claims overlap (the same point in different words, or one repeating part of another), state the point once in a single sentence that lists all of their IDs; never say the same thing twice. Claim IDs belong only in claim_ids, never in the text.
5. Citations go only in ref_keys, never in the text: no brackets, author names or years in the sentence. Use only citation keys shown next to the claims you are expressing.
6. Copy numbers exactly as written in the sources, including units and precision.
7. Every sentence states at least one claim. Do not write signposts or filler such as "This section describes ..." or "The following details are provided" (signpost is always false); create flow with connectives inside sentences that carry claims. Never restate what an earlier paragraph already said.
8. You decide wording, connectives, sentence order within the paragraph, tense and voice. Write connected prose, not a list: combine closely related claims, vary sentence structure, and use connectives such as "then", "in addition" or "whereas" where they fit. Use formal academic English: past tense for methods and results, present tense for established background, and "we" for the authors' own work. Claims typed in chat may be informal: express them in academic register ("cheap" becomes "low-cost"; drop asides such as "for now") without changing their meaning.

Return one paragraph as JSON. Each item in "sentences" holds exactly one sentence, with the claim IDs that sentence states."""


def _claim_line(c: Claim) -> str:
    refs = f" {{refs: {', '.join(c.refs)}}}" if c.refs else ""
    return f"[{c.id}] ({c.type.value}) {c.text.strip()}{refs}"


def _table_block(t: DataTable) -> str:
    lines = [f"{t.id}: {t.caption}", "  " + " | ".join(t.columns)]
    lines += ["  " + " | ".join(row) for row in t.rows]
    return "\n".join(lines)


EVIDENCE_CHARS_PER_CLAIM = 420
EVIDENCE_CHARS_TOTAL = 1500     # keeps a paragraph prompt inside a 4k-token local model


def _stems(text: str) -> set[str]:
    return {tu.stem(w) for w in tu.words(text) if w.lower() not in tu.STOPWORDS and len(w) > 2}


def focused_excerpt(claim: str, passage: str, max_chars: int = EVIDENCE_CHARS_PER_CLAIM) -> str:
    """The part of a passage that is about this claim: its best-matching line or sentence and the ones
    right next to it. A slide holds several claims; handing the writer the whole slide makes it restate
    sibling bullets that belong to other paragraphs."""
    units = [u.strip(" -•\t") for u in re.split(r"\n+|(?<=[.!?])\s+", passage) if len(u.strip(" -•\t")) > 3]
    if len(units) <= 1:
        return passage[:max_chars]
    target = _stems(claim)
    scores = [len(target & _stems(u)) / max(1, len(target)) for u in units]
    best = max(range(len(units)), key=scores.__getitem__)
    picked = [best]
    for j in (best + 1, best - 1):          # a neighbour that shares the claim's words usually continues it
        if 0 <= j < len(units) and scores[j] >= 0.3 and sum(len(units[k]) for k in picked) + len(units[j]) <= max_chars:
            picked.append(j)
    return " ".join(units[k] for k in sorted(picked))[:max_chars]


def writer_user_prompt(
    ledger: Ledger,
    section: str,
    claims: list[Claim],
    previous_paragraph: str = "",
    feedback: list[str] | None = None,
    evidence: dict[str, list[tuple[str, str]]] | None = None,
    style: str = "",
    target_words: int = 0,
) -> str:
    """evidence maps claim ID -> [(location, excerpt)] from the author's uploaded material or chat."""
    table_ids = {tid for c in claims for tid in c.tables}
    fig_ids = {fid for c in claims for fid in c.figures}
    tables = [t for t in ledger.tables if t.id in table_ids]
    figures = [f for f in ledger.figures if f.id in fig_ids]

    parts = [
        f"Paper title: {ledger.title}",
        f"Section: {section}",
        "",
        "Claims to express (all of them, nothing else):",
        *(_claim_line(c) for c in claims),
    ]
    if evidence:
        budget = EVIDENCE_CHARS_TOTAL
        lines = []
        for c in claims:
            for _loc, text in evidence.get(c.id, [])[:1]:
                if budget <= 0:
                    break
                snippet = re.sub(r"\s+", " ", focused_excerpt(c.text, text))[: min(EVIDENCE_CHARS_PER_CLAIM, budget)].strip()
                if not snippet or snippet.rstrip(".").lower() == c.text.rstrip(".").lower():
                    continue                        # adds nothing beyond the claim itself
                budget -= len(snippet)
                lines.append(f"{c.id}: \"{snippet}\"")
        if lines:
            parts += ["", "Supporting detail from the author's material, for the claim it is listed under only. "
                          "Never mention files, slides, pages or excerpts in the text:", *lines]
    if tables:
        parts += ["", "Tables referenced by these claims (you may refer to them as Table <n>):"]
        parts += [_table_block(t) for t in tables]
    if figures:
        parts += ["", "Figures referenced by these claims:"]
        parts += [f"{f.id}: {f.caption}" for f in figures]
    if previous_paragraph:
        # only its last sentence: shown a whole paragraph, a small model copies it into the new one
        last = (tu.split_sentences(previous_paragraph) or [previous_paragraph])[-1][:300]
        parts += ["", "The paragraph before this one ends with the sentence below. It is already written: "
                      "connect to it, but never restate it or anything else from earlier paragraphs.", last]
    if target_words:
        parts += ["", f"Length: about {target_words} words if the claims and excerpts support it. Never pad with content they do not contain; shorter is fine."]
    if style:
        parts += ["", f"The author's instruction for wording (style only, never new content): {style}"]
    if feedback:
        parts += [
            "",
            "Your previous attempt was rejected by the verifier for these reasons:",
            *(f"- {f}" for f in feedback),
            "Rewrite the paragraph so that every sentence is fully supported by the claims above. "
            "Remove anything that is not in the claims rather than rephrasing it.",
        ]
    return "\n".join(parts)


def writer_schema(claim_ids: list[str], ref_keys: list[str]) -> dict:
    """JSON schema whose enums restrict IDs to this paragraph's claims and citations."""
    claim_items: dict = {"type": "string", "enum": claim_ids} if claim_ids else {"type": "string"}
    ref_items: dict = {"type": "string", "enum": ref_keys} if ref_keys else {"type": "string"}
    ref_array: dict = {"type": "array", "items": ref_items}
    if not ref_keys:
        ref_array["maxItems"] = 0
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sentences"],
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "claim_ids", "ref_keys", "signpost"],
                    "properties": {
                        "text": {"type": "string"},
                        "claim_ids": {"type": "array", "items": claim_items},
                        "ref_keys": ref_array,
                        "signpost": {"type": "boolean"},
                    },
                },
            }
        },
    }
