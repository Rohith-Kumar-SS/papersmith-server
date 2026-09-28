"""Claim extraction from the author's own material, with a second check that nothing was invented.

The model reads a small group of passages (sized for a 4k-token local model) and lists the claims they
state. Every extracted claim is then checked against the passages it cites with the same rules the
verifier applies to prose: numbers must appear, new terms must appear, no added interpretation or
hedging, most of its wording must be in the passage, and (when available) the passage must entail it.
Claims that fail are dropped, so the ledger only ever holds what the author actually wrote.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import textutils as tu
from .llm import BackendError, LLMBackend, is_fatal
from .models import Claim, ClaimType, Ledger, Passage
from .nli import get_scorer

log = logging.getLogger(__name__)

CHUNK_CHARS = 2200
CHUNK_PASSAGES = 6
MIN_COVERAGE = 0.6
MIN_ENTAILMENT = 0.35

TYPES = [t.value for t in ClaimType]

OWN_SYSTEM = """You are a careful note-taker helping an author turn their own research material into a paper. List the factual claims the passages state, so they can be written up later. You are not a scientist here: never add knowledge, reasoning or interpretation of your own.

Rules:
1. Only what the passages state. Do not infer, generalise, interpret or explain.
2. One claim per item, written as a short self-contained note of 8 to 40 words. Copy numbers, units, names and technical terms exactly.
3. Keep the author's own hedges and interpretations if the passage has them ("may", "suggests"); never add any.
4. Give each claim one type: background (established knowledge about the field, never details of this study), gap (what prior work lacks), objective (what this work aims to do), contribution, method (how this study was done, including setup, equipment and procedure), dataset (the data this study collected or used, including where and how), result, comparison (a result against a baseline), interpretation (the author's reading of a result), limitation, future_work.
5. List the passage IDs each claim comes from.
6. Skip titles, agendas, section headings, thank-you slides, affiliations, contact details and reference lists.

Return JSON."""

REFERENCE_SYSTEM = """You are a careful note-taker. The passages come from a paper by other researchers that the author wants to cite. List what that paper does and reports, as short notes the author can use to describe it as prior work, for example "proposes a transformer model for pothole detection" or "reports an F1-score of 0.88 on the PotholeSet benchmark".

Rules:
1. Only what the passages state. Do not evaluate the paper or add knowledge of your own.
2. One claim per item, 8 to 40 words. Copy numbers, names and terms exactly.
3. Use type background for what the paper does or finds, and gap for limitations it states about earlier work.
4. List the passage IDs each claim comes from.
5. Skip the abstract's marketing phrases, author lists, affiliations, acknowledgements and the reference list.

Return JSON."""


def extraction_schema(passage_ids: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": ["claims"],
        "properties": {"claims": {"type": "array", "maxItems": 25, "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "type", "passages"],
            "properties": {
                "text": {"type": "string"},
                "type": {"type": "string", "enum": TYPES},
                "passages": {"type": "array", "items": {"type": "string", "enum": passage_ids}},
            },
        }}},
    }


def chunks(passages: list[Passage]) -> list[list[Passage]]:
    out: list[list[Passage]] = []
    current: list[Passage] = []
    size = 0
    for p in passages:
        if current and (size + len(p.text) > CHUNK_CHARS or len(current) >= CHUNK_PASSAGES):
            out.append(current)
            current, size = [], 0
        current.append(p)
        size += len(p.text)
    if current:
        out.append(current)
    return out


_HEADING_TYPES = [
    (re.compile(r"\b(data(set)?s?|materials?|participants|study (area|site)|collection|sampling)\b", re.I), ClaimType.dataset),
    (re.compile(r"\b(methods?|methodology|approach|set-?up|experiment(al)?|procedure|implementation|model|training)\b", re.I), ClaimType.method),
    (re.compile(r"\b(results?|evaluation|performance|findings|experiments and results)\b", re.I), ClaimType.result),
    (re.compile(r"\b(limitations?|threats to validity)\b", re.I), ClaimType.limitation),
    (re.compile(r"\b(future work|next steps|outlook)\b", re.I), ClaimType.future_work),
]


def _heading(p: Passage) -> str:
    """The slide title or document heading a passage sits under."""
    m = re.match(r"^§ (.+?) ¶", p.location)
    if m:
        return m.group(1)
    if p.location.startswith("slide"):
        return p.text.strip().splitlines()[0][:80] if p.text.strip() else ""
    return ""


def heading_type(p: Passage) -> ClaimType | None:
    h = _heading(p)
    for rx, ctype in _HEADING_TYPES:
        if h and rx.search(h):
            return ctype
    return None


def _prompt(filename: str, role: str, group: list[Passage]) -> str:
    what = "a paper by other authors, to cite" if role == "reference" else "the author's own work"
    lines = [f"Source: {filename} ({what})", "", "Passages:"]
    lines += [f"[{p.id}] ({p.location}) {p.text}" for p in group]
    lines += ["", "List the claims these passages state."]
    return "\n".join(lines)


# ---------------------------------------------------------------- the second check

@dataclass
class Checked:
    text: str
    type: ClaimType
    passages: list[str]
    ok: bool
    reason: str = ""


def check_claim(text: str, evidence: str) -> str:
    """Empty string when the evidence supports the claim, otherwise why it does not."""
    if len(tu.words(text)) < 3:
        return "too short"
    ev_numbers = tu.extract_numbers(evidence)
    for n in tu.extract_numbers(text):
        if not tu.number_supported(n, ev_numbers):
            return f"number {n:g} not in the passage"
    for term in tu.technical_terms(text):
        if not tu.term_in_source(term, evidence):
            return f"term '{term}' not in the passage"
    ev_markers = tu.markers(evidence)
    for cls in tu.markers(text):
        if cls not in ev_markers:
            return f"adds {cls} language"
    stems = {tu.stem(w) for w in tu.words(text) if w.lower() not in tu.STOPWORDS and len(w) > 2}
    if stems:
        coverage = len(stems & (tu.content_stems(evidence) | {tu.stem(w) for w in tu.GENERIC})) / len(stems)
        if coverage < MIN_COVERAGE:
            return f"only {coverage:.0%} of its wording is in the passage"
    return ""


def _nli_filter(items: list[Checked], evidence_of: dict[int, str]) -> None:
    scorer = get_scorer()
    if not scorer:
        return
    todo = [(i, c) for i, c in enumerate(items) if c.ok]
    scores = scorer.score([(evidence_of[i][:1800], c.text) for i, c in todo]) if todo else []
    for (i, c), (entail, contra) in zip(todo, scores):
        if contra >= 0.5 or entail < MIN_ENTAILMENT:
            c.ok = False
            c.reason = f"the passage does not entail it (NLI {entail:.2f})"


# ---------------------------------------------------------------- extraction

@dataclass
class ExtractionResult:
    accepted: list[Checked] = field(default_factory=list)
    rejected: list[Checked] = field(default_factory=list)
    failed_chunks: int = 0


def extract(backend: LLMBackend, filename: str, role: str, passages: list[Passage],
            on_progress=None, use_nli: bool = True) -> ExtractionResult:
    result = ExtractionResult()
    by_id = {p.id: p for p in passages}
    groups = chunks(passages)
    system = REFERENCE_SYSTEM if role == "reference" else OWN_SYSTEM
    for gi, group in enumerate(groups, start=1):
        if on_progress:
            on_progress(gi, len(groups))
        ids = [p.id for p in group]
        try:
            raw = backend.generate_json(system, _prompt(filename, role, group), extraction_schema(ids))
        except BackendError as exc:
            if is_fatal(exc):
                raise
            log.info("extraction chunk %d/%d failed: %s", gi, len(groups), exc)
            result.failed_chunks += 1
            continue
        items: list[Checked] = []
        evidence_of: dict[int, str] = {}
        for item in raw.get("claims", []):
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip().rstrip(".")
            pids = [p for p in item.get("passages", []) if p in by_id] or ids
            try:
                ctype = ClaimType(item.get("type", "background"))
            except ValueError:
                ctype = ClaimType.background
            if role == "reference" and ctype not in (ClaimType.background, ClaimType.gap):
                ctype = ClaimType.background
            elif role != "reference" and ctype == ClaimType.background:
                # "background" is the model's catch-all; a claim under a Data / Method / Results heading of the
                # author's own material describes this study, not the field
                ht = heading_type(by_id[pids[0]]) if pids and pids[0] in by_id else None
                if ht:
                    ctype = ht
            evidence = " ".join(by_id[p].text for p in pids)
            reason = check_claim(text, evidence)
            if reason and len(pids) < len(ids):          # the model may have cited the wrong passage
                wider = " ".join(p.text for p in group)
                if not check_claim(text, wider):
                    evidence, reason = wider, ""
                    pids = [p.id for p in group if tu.content_stems(p.text) & tu.content_stems(text)] or ids
            evidence_of[len(items)] = evidence
            items.append(Checked(text=text, type=ctype, passages=pids, ok=not reason, reason=reason))
        if use_nli:
            _nli_filter(items, evidence_of)
        for c in items:
            (result.accepted if c.ok else result.rejected).append(c)
    return result


# ---------------------------------------------------------------- tables

MAX_ROW_CLAIMS = 12


def table_row_claims(caption: str, columns: list[str], rows: list[list[str]], passage_id: str) -> list[Checked]:
    """One readable claim per table row, built without a model: the numbers are copied, never paraphrased."""
    if not rows or len(rows) > MAX_ROW_CLAIMS or len(columns) < 2:
        return []
    out = []
    subject = columns[0].strip()
    noun = f" {subject.lower()}" if subject and subject.lower() not in ("name", "series", "item", "row") else ""
    for row in rows:
        label = row[0].strip()
        values = [(c.strip(), v.strip()) for c, v in zip(columns[1:], row[1:]) if v.strip()]
        if not label or not values:
            continue
        parts = [f"{c} of {v}" if c else v for c, v in values]
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        # a plain sentence: it is also what the paper shows if the writer ever leaves the row out
        subject_words = f"{label}{noun}" if noun and noun.strip() not in label.lower() else label
        out.append(Checked(text=f"The {subject_words} had {joined}", type=ClaimType.result, passages=[passage_id], ok=True))
    return out


# ---------------------------------------------------------------- merging into the ledger

def _signature(text: str) -> set[str]:
    return {tu.stem(w) for w in tu.words(text) if w.lower() not in tu.STOPWORDS and len(w) > 2}


def next_claim_id(ledger: Ledger) -> str:
    nums = [int(c.id[1:]) for c in ledger.claims if c.id[1:].isdigit()]
    return f"C{(max(nums) if nums else 0) + 1}"


NLI_DUPLICATE = 0.8        # entailment above which one claim already says everything another does


def _family(t: ClaimType) -> str:
    return {"comparison": "result", "contribution": "objective"}.get(t.value, t.value)


def _nli_duplicate(ledger: Ledger, c: Checked, sig: set[str], nums: list[float]) -> tuple[str, Claim] | None:
    """("same", E) when existing claim E already says everything c says; ("absorb", E) when c says everything
    E says and more, so E can be folded into c. Word overlap misses paraphrases ("cheap" / "low-cost", a chat
    answer restating a slide); the entailment model catches them. Numbers must agree either way."""
    scorer = get_scorer()
    if not scorer or not sig or getattr(scorer, "remote", False):
        return None                 # an API judge would cost a request per new fact; word overlap still merges
    cands = []
    for e in ledger.claims:
        esig = _signature(e.text)
        if esig and _family(e.type) == _family(c.type) and len(sig & esig) / len(sig | esig) >= 0.25:
            cands.append(e)
    if not cands or not scorer.load():
        return None
    scores = scorer.score([(e.text, c.text) for e in cands] + [(c.text, e.text) for e in cands])
    n = len(cands)
    for i, e in enumerate(cands):
        if scores[i][0] >= NLI_DUPLICATE and all(tu.number_supported(x, tu.extract_numbers(e.text)) for x in nums):
            return "same", e
    for i, e in enumerate(cands):
        if scores[n + i][0] >= NLI_DUPLICATE and all(tu.number_supported(x, nums) for x in tu.extract_numbers(e.text)):
            return "absorb", e
    return None


def merge_into(ledger: Ledger, found: list[Checked], origin: str, refs: list[str] | None = None,
               tables: list[str] | None = None, nli: bool = False,
               absorbed: list[tuple[str, str]] | None = None) -> tuple[list[Claim], int]:
    """Add new claims; fold near-duplicates into the existing claim's sources. Returns (added, merged).

    With nli, paraphrased duplicates are caught too, and a new claim that says everything an older one says
    (and more) absorbs it: the older claim's receipts move to the new one and (old ID, new ID) is appended to
    `absorbed` so the caller can swap it in the outline."""
    added: list[Claim] = []
    merged = 0
    for c in found:
        sig = _signature(c.text)
        nums = sorted(tu.extract_numbers(c.text))
        dup = None
        for existing in ledger.claims:
            esig = _signature(existing.text)
            if not sig or not esig:
                continue
            jaccard = len(sig & esig) / len(sig | esig)
            if jaccard >= 0.75 and sorted(tu.extract_numbers(existing.text)) == nums:
                dup = existing
                break
        verdict = None if dup or not nli else _nli_duplicate(ledger, c, sig, nums)
        if verdict and verdict[0] == "same":
            dup = verdict[1]
        if dup:
            dup.sources = list(dict.fromkeys(dup.sources + c.passages))
            dup.refs = list(dict.fromkeys(dup.refs + (refs or [])))
            merged += 1
            continue
        claim = Claim(id=next_claim_id(ledger), type=c.type, text=c.text, refs=list(refs or []),
                      tables=list(tables or []), sources=list(c.passages), origin=origin)  # type: ignore[arg-type]
        if verdict and verdict[0] == "absorb":
            old = verdict[1]
            claim.sources = list(dict.fromkeys(old.sources + claim.sources))
            claim.refs = list(dict.fromkeys(old.refs + claim.refs))
            claim.tables = list(dict.fromkeys(old.tables + claim.tables))
            claim.figures = list(dict.fromkeys(old.figures + claim.figures))
            ledger.claims = [x for x in ledger.claims if x.id != old.id]
            added = [x for x in added if x.id != old.id]
            if absorbed is not None:
                absorbed.append((old.id, claim.id))
        ledger.claims.append(claim)
        added.append(claim)
    return added, merged


# ---------------------------------------------------------------- chat answers

def claims_from_answer(backend: LLMBackend | None, text: str, passage: Passage, default_type: ClaimType | None,
                       use_nli: bool = True) -> list[Checked]:
    """Claims from something the author typed. Short answers become one claim directly; longer ones are extracted."""
    clean = re.sub(r"\s+", " ", text).strip()
    words = tu.words(clean)
    if default_type and 3 <= len(words) <= 45 and len(tu.split_sentences(clean)) <= 2:
        return [Checked(text=clean.rstrip("."), type=default_type, passages=[passage.id], ok=True)]
    if backend is not None:
        try:
            res = extract(backend, "chat", "own", [passage], use_nli=use_nli)
            if res.accepted:
                if default_type:
                    for c in res.accepted:
                        if c.type == ClaimType.background:
                            c.type = default_type
                return res.accepted
        except BackendError:
            pass
    # fallback: every sentence the author wrote is a claim in their own words
    out = []
    for sent in tu.split_sentences(clean):
        if len(tu.words(sent)) >= 4:
            out.append(Checked(text=sent.rstrip("."), type=default_type or guess_type(sent), passages=[passage.id], ok=True))
    return out


def guess_type(text: str) -> ClaimType:
    t = text.lower()
    rules = [
        (r"\b(we (propose|present|introduce|release)|our contribution|contributions?)\b", ClaimType.contribution),
        (r"\b(we aim|aim(ed)? to|our goal|objective|this (work|study|paper) (investigates|examines|addresses))\b", ClaimType.objective),
        (r"\b(no (existing|prior)|has not been|have not been|lack of|remains? unclear|little is known)\b", ClaimType.gap),
        (r"\b(limitation|limited to|only (one|a single))\b", ClaimType.limitation),
        (r"\b(future work|we plan|next,? we will)\b", ClaimType.future_work),
        (r"\b(we attribute|suggests?|we believe|indicates?)\b", ClaimType.interpretation),
        (r"\b(compared (with|to)|than|versus|vs\.?|baseline|outperform)\b", ClaimType.comparison),
        (r"\b(dataset|data (were|was)|collected|participants|samples|recorded|images)\b", ClaimType.dataset),
        (r"\b(we (trained|used|applied|computed|measured|designed|implemented)|was trained|were (trained|segmented|computed))\b", ClaimType.method),
        (r"\b(achieved|reached|obtained|resulted|accuracy|f1|auc|precision|recall|error)\b", ClaimType.result),
    ]
    for pattern, ctype in rules:
        if re.search(pattern, t):
            return ctype
    return ClaimType.background
