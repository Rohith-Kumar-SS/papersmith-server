"""Deterministic outline planner.

Maps claims to IMRaD sections by claim type, keeping the author's order.
No model is involved: structure is an intellectual choice, so the planner
only proposes the conventional layout and the author approves or edits it.

Claims that describe papers the author uploaded to cite go to Related Work when
there are enough of them; otherwise they stay in the Introduction's background.
"""

from __future__ import annotations

from .models import Claim, ClaimType, Ledger, Outline, OutlineParagraph, OutlineSection

MAX_CLAIMS_PER_PARAGRAPH = 4
RELATED_WORK_MIN = 3

# (section, [groups of claim types]); each group starts a new paragraph run
SECTION_PLAN: list[tuple[str, list[list[ClaimType]]]] = [
    ("Introduction", [[ClaimType.background], [ClaimType.gap, ClaimType.objective], [ClaimType.contribution]]),
    ("Related Work", [[ClaimType.background, ClaimType.gap]]),
    ("Methods", [[ClaimType.dataset], [ClaimType.method]]),
    ("Results", [[ClaimType.result, ClaimType.comparison]]),
    ("Discussion", [[ClaimType.interpretation], [ClaimType.limitation]]),
    ("Conclusion", [[ClaimType.contribution, ClaimType.future_work]]),
]

SECTION_ORDER = ["Abstract"] + [name for name, _ in SECTION_PLAN]

# words that fit on one page of each template (text only; floats and references take the rest)
WORDS_PER_PAGE = {"ieee": 850, "acm": 800, "article": 500}
SECTION_SHARE = {"Introduction": 0.17, "Related Work": 0.14, "Methods": 0.24, "Results": 0.22,
                 "Discussion": 0.15, "Conclusion": 0.08}
ABSTRACT_WORDS = 180
WORDS_PER_CLAIM = 30          # what one claim typically becomes in prose, with its supporting detail


def _chunk(claims: list[Claim]) -> list[list[Claim]]:
    return [claims[i : i + MAX_CLAIMS_PER_PARAGRAPH] for i in range(0, len(claims), MAX_CLAIMS_PER_PARAGRAPH)]


def is_prior_work(c: Claim) -> bool:
    """A claim describing a cited paper the author uploaded (not the author's own framing)."""
    return c.origin == "file" and bool(c.refs) and c.type in (ClaimType.background, ClaimType.gap)


def _uses_related_work(ledger: Ledger) -> bool:
    return sum(1 for c in ledger.claims if is_prior_work(c)) >= RELATED_WORK_MIN


def section_for(c: Claim, ledger: Ledger) -> str:
    if c.type in (ClaimType.background, ClaimType.gap) and is_prior_work(c) and _uses_related_work(ledger):
        return "Related Work"
    for name, groups in SECTION_PLAN:
        if name == "Related Work":
            continue
        if any(c.type in types for types in groups):
            return name
    return "Introduction"


def _abstract_claims(ledger: Ledger) -> list[Claim]:
    by_type: dict[ClaimType, list[Claim]] = {}
    for c in ledger.claims:
        if not is_prior_work(c):
            by_type.setdefault(c.type, []).append(c)
    picks: list[Claim] = []
    picks += by_type.get(ClaimType.objective, [])[:1] or by_type.get(ClaimType.gap, [])[:1]
    picks += by_type.get(ClaimType.method, [])[:2]
    results = by_type.get(ClaimType.result, []) + by_type.get(ClaimType.comparison, [])
    # the author's prose results summarise better than individual table rows
    results.sort(key=lambda c: bool(c.sources) and all(".t" in s for s in c.sources))
    picks += results[:3]
    picks += by_type.get(ClaimType.contribution, [])[:1]
    return picks


def plan(ledger: Ledger, target_pages: int | None = None) -> Outline:
    sections: list[OutlineSection] = []
    counter = 0

    def para(claims: list[Claim]) -> OutlineParagraph:
        nonlocal counter
        counter += 1
        return OutlineParagraph(id=f"P{counter}", claim_ids=[c.id for c in claims])

    abstract = _abstract_claims(ledger)
    if abstract:
        sections.append(OutlineSection(name="Abstract", paragraphs=[para(abstract)]))

    for name, groups in SECTION_PLAN:
        paragraphs = []
        for types in groups:
            group = [c for c in ledger.claims if c.type in types and section_for(c, ledger) == name] \
                if name in ("Introduction", "Related Work") else [c for c in ledger.claims if c.type in types]
            if name == "Conclusion":
                # conclusion restates at most two contributions plus all future work
                contrib = [c for c in group if c.type == ClaimType.contribution][:2]
                group = contrib + [c for c in group if c.type == ClaimType.future_work]
            paragraphs.extend(para(chunk) for chunk in _chunk(group))
        if paragraphs:
            sections.append(OutlineSection(name=name, paragraphs=paragraphs))

    outline = Outline(sections=sections, approved=False)
    if target_pages:
        apply_targets(outline, ledger.template, target_pages)
    return outline


# ---------------------------------------------------------------- length

def words_per_page(template: str) -> int:
    return WORDS_PER_PAGE.get(template, 700)


def apply_targets(outline: Outline, template: str, target_pages: int) -> None:
    body = max(0, int(target_pages * words_per_page(template) * 0.85) - ABSTRACT_WORDS)
    present = [s for s in outline.sections if s.name in SECTION_SHARE]
    total_share = sum(SECTION_SHARE[s.name] for s in present) or 1
    for s in outline.sections:
        s.target_words = ABSTRACT_WORDS if s.name == "Abstract" else int(body * SECTION_SHARE.get(s.name, 0.1) / total_share)


def supported_words(ledger: Ledger, outline: Outline | None = None) -> int:
    """How much faithful prose the material supports: roughly WORDS_PER_CLAIM per placed claim."""
    ids = {cid for s in outline.sections if s.name != "Abstract" for p in s.paragraphs for cid in p.claim_ids} if outline \
        else {c.id for c in ledger.claims}
    return ABSTRACT_WORDS + WORDS_PER_CLAIM * len(ids)


def paragraph_target(section: OutlineSection) -> int:
    n = max(1, len(section.paragraphs))
    return max(60, min(260, section.target_words // n)) if section.target_words else 0


# ---------------------------------------------------------------- incremental changes

def _next_paragraph_id(outline: Outline) -> str:
    nums = [int(p.id[1:]) for s in outline.sections for p in s.paragraphs if p.id[1:].isdigit()]
    return f"P{(max(nums) if nums else 0) + 1}"


def place_claim(outline: Outline, ledger: Ledger, claim: Claim) -> str:
    """Put a new claim into the outline without reshuffling what is written. Returns the paragraph ID."""
    name = section_for(claim, ledger)
    section = next((s for s in outline.sections if s.name == name), None)
    if section is None:
        section = OutlineSection(name=name)
        order = {n: i for i, n in enumerate(SECTION_ORDER)}
        at = next((i for i, s in enumerate(outline.sections) if order.get(s.name, 99) > order[name]), len(outline.sections))
        outline.sections.insert(at, section)
    target = section.paragraphs[-1] if section.paragraphs else None
    if target is None or len(target.claim_ids) >= MAX_CLAIMS_PER_PARAGRAPH:
        target = OutlineParagraph(id=_next_paragraph_id(outline), claim_ids=[])
        section.paragraphs.append(target)
    target.claim_ids.append(claim.id)
    target.stale = True
    return target.id


def replace_claim(outline: Outline, old_id: str, new_id: str) -> bool:
    """Put new_id where old_id sits (a more specific claim absorbed a vaguer one). True if it was placed."""
    placed = False
    for s in outline.sections:
        for p in s.paragraphs:
            if old_id in p.claim_ids:
                i = p.claim_ids.index(old_id)
                if new_id in p.claim_ids or placed:
                    p.claim_ids.pop(i)
                else:
                    p.claim_ids[i] = new_id
                    placed = True
                p.stale = True
        s.paragraphs = [p for p in s.paragraphs if p.claim_ids]
    outline.sections = [s for s in outline.sections if s.paragraphs]
    return placed


def remove_claim(outline: Outline, claim_id: str) -> list[str]:
    """Take a claim out of every paragraph; returns the paragraphs that changed."""
    changed = []
    for s in outline.sections:
        for p in s.paragraphs:
            if claim_id in p.claim_ids:
                p.claim_ids.remove(claim_id)
                p.stale = True
                changed.append(p.id)
        s.paragraphs = [p for p in s.paragraphs if p.claim_ids]
    outline.sections = [s for s in outline.sections if s.paragraphs]
    return changed


def refresh_abstract(outline: Outline, ledger: Ledger) -> bool:
    """Keep the abstract's claims in step with the ledger; True when it changed."""
    wanted = [c.id for c in _abstract_claims(ledger)]
    section = next((s for s in outline.sections if s.name == "Abstract"), None)
    if not wanted:
        return False
    if section is None:
        outline.sections.insert(0, OutlineSection(name="Abstract", target_words=ABSTRACT_WORDS,
                                                  paragraphs=[OutlineParagraph(id=_next_paragraph_id(outline), claim_ids=wanted, stale=True)]))
        return True
    para = section.paragraphs[0]
    if para.claim_ids != wanted:
        para.claim_ids = wanted
        para.stale = True
        return True
    return False


def validate(outline: Outline, ledger: Ledger) -> list[str]:
    """Problems with a (possibly user-edited) outline."""
    problems = []
    known = ledger.claim_map()
    used = set()
    for section in outline.sections:
        for p in section.paragraphs:
            if not p.claim_ids:
                problems.append(f"{section.name}/{p.id}: paragraph has no claims")
            for cid in p.claim_ids:
                if cid not in known:
                    problems.append(f"{section.name}/{p.id}: unknown claim {cid}")
                used.add(cid)
    for cid in known:
        if cid not in used:
            problems.append(f"claim {cid} is not placed in any paragraph")
    return problems
