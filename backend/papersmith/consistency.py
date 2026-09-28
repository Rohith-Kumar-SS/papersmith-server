"""Whole-document consistency checks. Reports problems; never proposes fixes or new claims."""

from __future__ import annotations

import itertools
import re

from . import textutils as tu
from .models import ClaimType, ConsistencyIssue, Draft, Ledger
from .nli import get_scorer

METRIC_WORDS = r"accuracy|auc|auroc|f1|f1-score|precision|recall|sensitivity|specificity|kappa|mae|mse|rmse|" \
               r"bleu|rouge|dice|iou|map|error|loss|r2|correlation|p-value|odds ratio|hazard ratio"


def _table_numbers(ledger: Ledger, table_ids: list[str]) -> list[float]:
    tables = ledger.table_map()
    nums: list[float] = []
    for tid in table_ids:
        t = tables.get(tid)
        if t:
            nums += tu.extract_numbers(" ".join(" ".join(r) for r in t.rows), include_words=False)
    return nums


def check_ledger(ledger: Ledger, use_nli: bool = True) -> list[ConsistencyIssue]:
    issues: list[ConsistencyIssue] = []
    refs = ledger.ref_map()
    tables = ledger.table_map()
    figures = {f.id for f in ledger.figures}

    seen: dict[str, str] = {}
    for c in ledger.claims:
        # dangling links
        for r in c.refs:
            if r not in refs:
                issues.append(ConsistencyIssue(kind="missing_reference", severity="error",
                                               detail=f"{c.id} cites '{r}', which is not in the reference library", locations=[c.id]))
        for t in c.tables:
            if t not in tables:
                issues.append(ConsistencyIssue(kind="missing_table", severity="error",
                                               detail=f"{c.id} links table {t}, which does not exist", locations=[c.id]))
        for f in c.figures:
            if f not in figures:
                issues.append(ConsistencyIssue(kind="missing_figure", severity="error",
                                               detail=f"{c.id} links figure {f}, which does not exist", locations=[c.id]))

        # claim numbers vs linked tables  (e.g. claim says 92% but the table says 89.0)
        if c.tables:
            table_nums = _table_numbers(ledger, c.tables)
            for n in tu.extract_numbers(c.text, include_words=False):
                if table_nums and not tu.number_supported(n, table_nums):
                    shown = int(n) if n.is_integer() else n
                    issues.append(ConsistencyIssue(kind="claim_table_mismatch", severity="warning",
                                                   detail=f"{c.id} states {shown}, which does not appear in {', '.join(c.tables)}",
                                                   locations=[c.id, *c.tables]))

        # background claims usually need a citation
        if c.type == ClaimType.background and not c.refs:
            issues.append(ConsistencyIssue(kind="uncited_background", severity="info",
                                           detail=f"{c.id} is background knowledge with no citation attached", locations=[c.id]))

        key = tu.normalize(c.text)
        if key in seen:
            issues.append(ConsistencyIssue(kind="duplicate_claim", severity="info",
                                           detail=f"{c.id} duplicates {seen[key]}", locations=[seen[key], c.id]))
        seen.setdefault(key, c.id)

    # unused library items
    used_refs = {r for c in ledger.claims for r in c.refs}
    for r in ledger.references:
        if r.key not in used_refs:
            issues.append(ConsistencyIssue(kind="unused_reference", severity="info",
                                           detail=f"reference '{r.key}' is not attached to any claim", locations=[r.key]))
    used_tables = {t for c in ledger.claims for t in c.tables}
    for t in ledger.tables:
        if t.id not in used_tables:
            issues.append(ConsistencyIssue(kind="unused_table", severity="info",
                                           detail=f"table {t.id} is not linked to any claim", locations=[t.id]))

    # possibly contradictory result claims (NLI both directions)
    if use_nli:
        scorer = get_scorer()
        pool = [c for c in ledger.claims if c.type in (ClaimType.result, ClaimType.comparison, ClaimType.interpretation)][:20]
        pairs = list(itertools.combinations(pool, 2))
        if scorer and pairs:
            scores = scorer.score([(a.text, b.text) for a, b in pairs] + [(b.text, a.text) for a, b in pairs])
            if scores:
                n = len(pairs)
                for i, (a, b) in enumerate(pairs):
                    # both directions must agree: NLI often calls two different numbers for different methods a contradiction
                    contra = min(scores[i][1], scores[i + n][1])
                    if contra >= 0.8:
                        issues.append(ConsistencyIssue(kind="possible_contradiction", severity="warning",
                                                       detail=f"{a.id} and {b.id} may contradict each other (NLI p={contra:.2f})",
                                                       locations=[a.id, b.id]))
    return issues


def check_draft(ledger: Ledger, draft: Draft) -> list[ConsistencyIssue]:
    issues: list[ConsistencyIssue] = []

    # coverage
    expressed = {cid for p in draft.paragraphs for s in p.sentences for cid in s.claim_ids}
    for c in ledger.claims:
        if c.id not in expressed:
            issues.append(ConsistencyIssue(kind="claim_missing_from_draft", severity="warning",
                                           detail=f"{c.id} is not expressed anywhere in the draft", locations=[c.id]))
    cited = {k for p in draft.paragraphs for s in p.sentences for k in s.ref_keys}
    for r in ledger.references:
        if r.key not in cited:
            issues.append(ConsistencyIssue(kind="reference_never_cited", severity="info",
                                           detail=f"reference '{r.key}' is never cited in the draft", locations=[r.key]))

    # abstract / conclusion numbers must be backed by the body
    body_numbers: list[float] = []
    for p in draft.paragraphs:
        if p.section not in ("Abstract", "Conclusion"):
            for s in p.sentences:
                body_numbers += tu.extract_numbers(s.text, include_words=False)
    body_numbers += _table_numbers(ledger, [t.id for t in ledger.tables])
    for p in draft.paragraphs:
        if p.section in ("Abstract", "Conclusion"):
            for s in p.sentences:
                for n in tu.extract_numbers(s.text, include_words=False):
                    if not tu.number_supported(n, body_numbers):
                        shown = int(n) if n.is_integer() else n
                        issues.append(ConsistencyIssue(kind="summary_number_unbacked", severity="warning",
                                                       detail=f"{p.section} states {shown}, which never appears in the body or tables",
                                                       locations=[s.id]))

    # the same metric reported with different values
    metric_re = re.compile(rf"\b({METRIC_WORDS})\b[^.;]{{0,40}}?(\d+(?:\.\d+)?)\s?%?", re.I)
    by_metric: dict[str, dict[str, list[str]]] = {}
    for p in draft.paragraphs:
        for s in p.sentences:
            for m in metric_re.finditer(s.text):
                by_metric.setdefault(m.group(1).lower(), {}).setdefault(m.group(2), []).append(s.id)
    abstract_ids = {s.id for p in draft.paragraphs if p.section == "Abstract" for s in p.sentences}
    for metric, values in by_metric.items():
        abstract_vals = {v for v, ids in values.items() if set(ids) & abstract_ids}
        for v in abstract_vals:
            others = {ov for ov in values if ov != v}
            body_reports_v = bool(set(values[v]) - abstract_ids)
            if others and not body_reports_v:
                issues.append(ConsistencyIssue(kind="metric_value_mismatch", severity="warning",
                                               detail=f"Abstract reports {metric} = {v}, the body reports {', '.join(sorted(others))}",
                                               locations=values[v] + [i for ov in others for i in values[ov]]))

    # acronyms used without being defined
    text = " ".join(s.text for p in draft.paragraphs for s in p.sentences)
    defined = set(re.findall(r"\(([A-Z][A-Za-z0-9-]*[A-Z0-9]s?)\)", text))
    for acr in sorted({t for t in re.findall(r"\b[A-Z]{2,}[0-9]*\b", text)}):
        if acr not in defined and acr not in {"AI", "US", "UK", "EU", "DNA", "RNA", "USA"}:
            issues.append(ConsistencyIssue(kind="acronym_undefined", severity="info",
                                           detail=f"acronym '{acr}' is used but never defined in parentheses", locations=[]))
    return issues


def check_all(ledger: Ledger, draft: Draft | None, use_nli: bool = True) -> list[ConsistencyIssue]:
    issues = check_ledger(ledger, use_nli)
    if draft:
        issues += check_draft(ledger, draft)
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda i: order[i.severity])
