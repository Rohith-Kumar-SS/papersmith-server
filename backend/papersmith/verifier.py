"""Sentence-level verification: the real guarantee against ideation.

Each sentence is checked against the ledger items it claims to express:
  - traceability      (claim IDs present and valid)
  - numbers           (every number appears in the sources)
  - citations         (keys exist in the author's library and belong to the claims; no inline citations)
  - new terms         (acronyms / names / identifiers absent from the sources)
  - added content     (interpretive, causal, hedging, evaluative, comparative, future-work language not in sources)
  - entailment (NLI)  (the sources entail the sentence and do not contradict it)
"""

from __future__ import annotations

import re

from . import textutils as tu
from .config import settings
from .models import DraftParagraph, Flag, Ledger, Sentence
from .nli import get_scorer

MARKER_MESSAGES = {
    "interpretation": "adds an interpretation",
    "causal": "adds a causal claim",
    "hedge": "adds hedging/speculation",
    "evaluative": "adds an evaluative judgement",
    "comparative": "adds a comparison",
    "inference": "adds an inference or consequence",
    "future": "adds a future-work idea",
}


def _linearize_table(ledger: Ledger, table_id: str) -> str:
    t = ledger.table_map().get(table_id)
    if not t:
        return ""
    rows = ["; ".join(f"{col} = {val}" for col, val in zip(t.columns, row)) for row in t.rows]
    return f"Table {_digits(t.id)} ({t.caption}): " + ". ".join(rows) + "."


def _digits(item_id: str) -> str:
    return re.sub(r"\D", "", item_id) or item_id


def source_text(ledger: Ledger, claim_ids: list[str]) -> str:
    claims = ledger.claim_map()
    figs = {f.id: f for f in ledger.figures}
    parts = []
    for cid in claim_ids:
        c = claims.get(cid)
        if not c:
            continue
        parts.append(c.text.strip().rstrip(".") + ".")
        parts += [_linearize_table(ledger, tid) for tid in c.tables]
        parts += [f"Figure {_digits(fid)}: {figs[fid].caption}." for fid in c.figures if fid in figs]
    return " ".join(p for p in parts if p)


CLAIM_COVERAGE_MIN = 0.5
PARAPHRASE_ENTAILMENT = 0.7     # the text entails the claim: a paraphrase, not a claim left out


def claim_coverage(claim: str, text: str) -> float:
    """Share of the claim's content words (by stem) that appear in the text."""
    claim_stems = {tu.stem(w) for w in tu.words(claim) if w.lower() not in tu.STOPWORDS and len(w) > 2}
    if not claim_stems:
        return 1.0
    return len(claim_stems & tu.content_stems(text)) / len(claim_stems)


_LEADING_CONNECTIVE = re.compile(
    r"^(additionally|furthermore|moreover|in addition|also|then|next|finally|first|second|third|however|"
    r"in contrast|by contrast|similarly|likewise|specifically|in particular|overall|in summary|to this end)\s*,?\s+", re.I)


_FLOAT_POINTER = re.compile(r",?\s*\(?(?:as )?(?:shown|reported|listed|summari[sz]ed|presented|given|seen|see)\s+in\s+"
                            r"(?:Table|Tab\.|Figure|Fig\.)\s*\d+\)?|\s*\((?:see )?(?:Table|Tab\.|Figure|Fig\.)\s*\d+\)", re.I)


def _nli_hypothesis(text: str) -> str:
    """Sentence as NLI hypothesis: citations, pointers such as "as shown in Table 1", and content-free leading
    connectives removed (none carries a claim, and all of them confuse NLI models)."""
    s = _strip_citations(text)
    s = _FLOAT_POINTER.sub("", s)
    s = _LEADING_CONNECTIVE.sub("", s)
    return s[:1].upper() + s[1:]


def repeats(text: str, earlier: str, threshold: float = 0.7) -> bool:
    """True when a sentence says again what an earlier one said: nearly the same words and no new
    numbers, or the whole earlier sentence copied in with something added (the local model's habit
    when it is shown the previous paragraph)."""
    mine = tu.content_stems(_strip_citations(text))
    theirs = tu.content_stems(_strip_citations(earlier))
    if len(mine) < 5 or not theirs:
        return False
    shared = len(mine & theirs)
    my_numbers, their_numbers = tu.extract_numbers(text), tu.extract_numbers(earlier)
    if len(theirs) >= 6 and shared / len(theirs) >= 0.8 and all(tu.number_supported(n, my_numbers) for n in their_numbers):
        return True
    new_numbers = [n for n in my_numbers if not tu.number_supported(n, their_numbers)]
    return not new_numbers and shared / len(mine | theirs) >= threshold


def flag_repeats(paragraph: DraftParagraph, others: list[DraftParagraph], threshold: float = 0.7) -> None:
    """Warn when a sentence restates one already written in the paper; `others` are the paragraphs
    before this one (the abstract and conclusion are allowed to summarise). Safe to run again."""
    for s in paragraph.sentences:
        if any(f.kind == "repeats_earlier" for f in s.flags):
            s.flags = [f for f in s.flags if f.kind != "repeats_earlier"]
            set_status(s)
    if paragraph.section in ("Abstract", "Conclusion"):
        return
    earlier = [s for o in others if o.id != paragraph.id and o.section not in ("Abstract", "Conclusion") for s in o.sentences]
    for s in paragraph.sentences:
        if s.signpost:
            continue
        for o in earlier:
            if repeats(s.text, o.text, threshold):
                s.flags.append(Flag(kind="repeats_earlier", severity="warning",
                                    detail=f"repeats {o.id}, which already says this; leave that content out here"))
                set_status(s)
                break


def _strip_citations(text: str) -> str:
    out = text
    for hit in tu.inline_citations(text):
        out = out.replace(hit, "")
    return re.sub(r"\s+([.,;])", r"\1", out).strip()


Evidence = dict[str, list[tuple[str, str]]]      # claim ID -> [(location, excerpt of the author's material)]


def evidence_text(evidence: Evidence | None, claim_ids: list[str]) -> str:
    if not evidence:
        return ""
    return " ".join(text for cid in claim_ids for _loc, text in evidence.get(cid, []))


_QUANTITY_COMPARISON = re.compile(r"\b(over|above|more than|less than|below|under|greater|fewer|than|exceed\w*|up to)\b", re.I)


def check_sentence(sentence: Sentence, paragraph_claims: list[str], ledger: Ledger, section: str,
                   evidence: Evidence | None = None, nli_active: bool = False) -> list[Flag]:
    flags: list[Flag] = []
    known = ledger.claim_map()
    refs = ledger.ref_map()
    text = sentence.text.strip()

    # ---- traceability
    for cid in sentence.claim_ids:
        if cid not in known:
            flags.append(Flag(kind="unknown_claim", severity="error", detail=f"cites claim {cid}, which does not exist"))
        elif cid not in paragraph_claims:
            flags.append(Flag(kind="claim_outside_paragraph", severity="warning",
                              detail=f"expresses {cid}, which the outline assigns to another paragraph"))
    if not sentence.signpost and not sentence.claim_ids:
        flags.append(Flag(kind="untraced", severity="error", detail="sentence is not linked to any author claim"))

    valid_ids = [c for c in sentence.claim_ids if c in known]
    # a sentence may use detail from the excerpts of the claims it cites: that is the author's own material
    source = " ".join(x for x in (source_text(ledger, valid_ids), evidence_text(evidence, valid_ids)) if x)
    paragraph_source = " ".join(x for x in (source_text(ledger, paragraph_claims), evidence_text(evidence, paragraph_claims)) if x)
    term_source = " ".join([source if not sentence.signpost else paragraph_source, ledger.title, " ".join(ledger.keywords), section])
    analysed = _strip_citations(text)

    # ---- numbers
    source_numbers = tu.extract_numbers(source) if not sentence.signpost else []
    for n in tu.extract_numbers(analysed):
        if not tu.number_supported(n, source_numbers):
            shown = int(n) if n.is_integer() else n
            flags.append(Flag(kind="number_not_in_source", severity="error",
                              detail=f"number {shown} does not appear in the supporting claims or tables"))

    # ---- citations
    for hit in tu.inline_citations(text):
        if hit.lower() not in source.lower():
            flags.append(Flag(kind="inline_citation", severity="error",
                              detail=f"writes a citation in the text ('{hit}'); citations must come from the library"))
    attached = {r for cid in valid_ids for r in known[cid].refs}
    for key in sentence.ref_keys:
        if key not in refs:
            flags.append(Flag(kind="invented_citation", severity="error", detail=f"cites '{key}', which is not in the reference library"))
        elif key not in attached:
            flags.append(Flag(kind="citation_not_attached", severity="warning",
                              detail=f"cites '{key}', which the author did not attach to the claims in this sentence"))
    if sentence.signpost and sentence.ref_keys:
        flags.append(Flag(kind="signpost_carries_content", severity="error", detail="signpost sentence carries citations"))

    # ---- new terms (acronyms, identifiers, proper nouns)
    for term in tu.technical_terms(analysed) + tu.proper_nouns(analysed):
        if not tu.term_in_source(term, term_source):
            flags.append(Flag(kind="new_term", severity="warning" if not sentence.signpost else "error",
                              detail=f"introduces '{term}', which does not appear in the author's claims"))

    # ---- added content markers
    source_markers = tu.markers(source) if not sentence.signpost else {}
    if not sentence.signpost and _QUANTITY_COMPARISON.search(source):
        source_markers.setdefault("comparative", [])     # "cost over 10,000 USD" supports "exceeds 10,000 USD"
    for cls, hits in tu.markers(analysed).items():
        if cls not in source_markers:
            flags.append(Flag(kind=f"added_{cls}", severity="warning",
                              detail=f"{MARKER_MESSAGES[cls]} ('{', '.join(dict.fromkeys(h.lower() for h in hits))}') not present in the claims"))

    # ---- unsupported wording (soft signal; paraphrase is allowed). When the entailment model runs it judges
    # meaning, so unfamiliar words ("selected" for "chosen") are only a note; without it they are a warning.
    novel = tu.novel_content_words(analysed, term_source)
    content = [w for w in tu.words(analysed) if w.lower() not in tu.STOPWORDS and len(w) > 2]
    if novel:
        ratio = len(novel) / max(1, len(content))
        strong = len(novel) >= 2 and ratio > 0.15
        severity = "warning" if strong and not nli_active else "info"
        flags.append(Flag(kind="unsupported_wording", severity=severity,
                          detail=f"wording not found in the claims: {', '.join(novel[:8])}"))

    # ---- signpost hygiene
    if sentence.signpost and len(tu.words(analysed)) > 30:
        flags.append(Flag(kind="signpost_too_long", severity="warning", detail="signpost sentences should be short and content-free"))

    return flags


def verify_paragraph(paragraph: DraftParagraph, ledger: Ledger, use_nli: bool = True,
                     evidence: Evidence | None = None) -> DraftParagraph:
    known = ledger.claim_map()
    scorer = get_scorer() if use_nli else None
    nli_active = bool(scorer) and scorer.load()
    for s in paragraph.sentences:
        if s.signpost and s.claim_ids:      # older drafts: a sentence with claims is checked as content
            s.signpost = False
        s.flags = check_sentence(s, paragraph.claim_ids, ledger, paragraph.section, evidence, nli_active)
        s.entailment = s.contradiction = None

    # ---- NLI, batched per paragraph
    targets = [s for s in paragraph.sentences if not s.signpost and s.claim_ids]
    if scorer and targets:
        # premise = the claim texts plus their prose excerpts. Linearised tables confuse NLI models and stay
        # out; numbers are checked exactly above.
        def premise(s: Sentence) -> str:
            claims_text = " ".join(known[c].text.strip().rstrip(".") + "." for c in s.claim_ids if c in known)
            extra = evidence_text(evidence, [c for c in s.claim_ids if c in known])[:900]
            return f"{claims_text} {extra}".strip()

        pairs = [(premise(s), _nli_hypothesis(s.text)) for s in targets]
        scores = scorer.score(pairs)
        # a sentence may lean on what the paper has already established ("the low-cost sensors" when only the
        # title says low-cost): when its own claims alone look weak, judge it against the title and the rest of
        # the paragraph's claims too. Contradiction stays judged against the sentence's own claims.
        weak = [i for i, (e, c) in enumerate(scores) if e < settings.entail_threshold and c < settings.contradiction_threshold]
        if weak:
            context = " ".join(known[c].text.strip().rstrip(".") + "." for c in paragraph.claim_ids if c in known)
            wider = scorer.score([(f"{ledger.title}. {context} {pairs[i][0]}"[:2400], pairs[i][1]) for i in weak])
            for i, (e, _c) in zip(weak, wider):
                scores[i] = (max(scores[i][0], e), scores[i][1])
        for s, (entail, contra) in zip(targets, scores):
            s.entailment, s.contradiction = round(entail, 3), round(contra, 3)
            if contra >= settings.contradiction_threshold:
                s.flags.append(Flag(kind="contradicts_source", severity="error",
                                    detail=f"NLI: the claims contradict this sentence (p={contra:.2f})"))
            elif entail < settings.entail_threshold:
                s.flags.append(Flag(kind="weak_entailment", severity="warning",
                                    detail=f"NLI: the claims do not clearly entail this sentence (p={entail:.2f})"))

    # ---- paragraph-level coverage
    issues: list[Flag] = []
    if not paragraph.sentences:
        issues.append(Flag(kind="empty_paragraph", severity="error", detail="no sentences were produced"))
    # reverse check: a claim tagged on a sentence must actually be stated there, not just receipted. Few shared
    # words is fine when the entailment model confirms the text states the claim (good paraphrase: "cheap" ->
    # "low-cost", two overlapping claims merged into one sentence).
    low: dict[str, tuple[str, float]] = {}
    for cid in paragraph.claim_ids:
        expressing = [s for s in paragraph.sentences if cid in s.claim_ids]
        if expressing and cid in known:
            said = " ".join(_strip_citations(s.text) for s in expressing)
            coverage = claim_coverage(known[cid].text, said)
            if coverage < CLAIM_COVERAGE_MIN:
                low[cid] = (said, coverage)
    if low and scorer and nli_active:
        ids = list(low)
        scores = scorer.score([(_nli_hypothesis(low[c][0]), known[c].text) for c in ids])
        for cid, (entail, _contra) in zip(ids, scores):
            if entail >= PARAPHRASE_ENTAILMENT:
                del low[cid]
    for cid in paragraph.claim_ids:
        expressing = [s for s in paragraph.sentences if cid in s.claim_ids]
        if not expressing:
            issues.append(Flag(kind="claim_not_expressed", severity="warning", detail=f"claim {cid} is not expressed in this paragraph"))
            continue
        if cid in known:
            said = " ".join(_strip_citations(s.text) for s in expressing)
            coverage = low[cid][1] if cid in low else 1.0
            if cid in low:
                for s in expressing:
                    s.flags.append(Flag(kind="claim_underexpressed", severity="warning",
                                        detail=f"tagged with {cid}, but only {coverage:.0%} of that claim's content appears in the text"))
            for n in tu.extract_numbers(known[cid].text, include_words=False):
                if not tu.number_supported(n, tu.extract_numbers(said)):
                    shown = int(n) if n.is_integer() else n
                    issues.append(Flag(kind="claim_number_dropped", severity="warning",
                                       detail=f"claim {cid} reports {shown}, which the text expressing it leaves out"))
        cited = {k for s in expressing for k in s.ref_keys}
        for ref in known[cid].refs if cid in known else []:
            if ref not in cited:
                issues.append(Flag(kind="citation_dropped", severity="warning",
                                   detail=f"claim {cid} is supported by '{ref}' but the text does not cite it"))
    paragraph.issues = issues

    for s in paragraph.sentences:
        set_status(s)
    return paragraph


def set_status(s: Sentence) -> None:
    if s.status in ("user_edited", "accepted"):
        return
    s.status = "flagged" if any(f.severity in ("error", "warning") for f in s.flags) else "verified"


def paragraph_score(p: DraftParagraph) -> int:
    """Lower is better: 3 per error, 1 per warning."""
    flags = [f for s in p.sentences for f in s.flags] + p.issues
    return sum(3 if f.severity == "error" else 1 if f.severity == "warning" else 0 for f in flags)


def feedback_for(p: DraftParagraph) -> list[str]:
    lines = []
    for s in p.sentences:
        for f in s.flags:
            if f.severity in ("error", "warning"):
                lines.append(f'"{s.text[:90]}" - {f.detail}')
    lines += [f.detail for f in p.issues if f.severity in ("error", "warning")]
    return lines
