"""Provenance statistics and the AI-use disclosure statement.

The disclosure text is a fixed template filled with measured numbers; no model writes it.
"""

from __future__ import annotations

from collections import Counter

from . import __version__
from .models import Project, now_iso


def stats(project: Project) -> dict:
    draft = project.draft
    ledger = project.ledger
    sentences = [s for p in draft.paragraphs for s in p.sentences] if draft else []
    flags = Counter(f.kind for s in sentences for f in s.flags if f.severity != "info")
    status = Counter(s.status for s in sentences)
    # author edits are re-verified too: an edited sentence with open flags still needs review
    needs_review = [s for s in sentences if s.status != "accepted" and any(f.severity in ("error", "warning") for f in s.flags)]
    traced = [s for s in sentences if s.claim_ids]
    signposts = [s for s in sentences if s.signpost]
    human = [s for s in sentences if s.origin == "human" or s.status == "user_edited"]
    entail = [s.entailment for s in sentences if s.entailment is not None]
    expressed = {cid for s in sentences for cid in s.claim_ids}
    cited = {k for s in sentences for k in s.ref_keys}
    library = {r.key for r in ledger.references}
    total = len(sentences) or 1
    return {
        "generated_at": now_iso(),
        "tool": f"PaperSmithAI {__version__}",
        "backend": draft.backend if draft else "",
        "model": draft.model if draft else "",
        "claims": len(ledger.claims),
        "claims_expressed": len(expressed & {c.id for c in ledger.claims}),
        "references": len(library),
        "sentences": len(sentences),
        "traced_sentences": len(traced),
        "signpost_sentences": len(signposts),
        "traceability_pct": round(100 * (len(traced) + len(signposts)) / total, 1),
        "verified": status.get("verified", 0),
        "flagged": len(needs_review),
        "user_edited": len(human),
        "accepted_after_review": status.get("accepted", 0),
        "flag_counts": dict(flags.most_common()),
        "leakage_rate_pct": round(100 * sum(1 for s in sentences if any(f.severity == "error" for f in s.flags)) / total, 1),
        "invented_citations": flags.get("invented_citation", 0) + flags.get("inline_citation", 0),
        "citations_outside_library": len(cited - library),
        "mean_entailment": round(sum(entail) / len(entail), 3) if entail else None,
        "rewrite_attempts": sum(p.attempts for p in draft.paragraphs) if draft else 0,
    }


def disclosure_statement(project: Project) -> str:
    st = stats(project)
    model = f"{st['model']} via {st['backend']}" if st["backend"] else "no model"
    return (
        f"The prose of this manuscript was drafted with {st['tool']} ({model}). "
        f"All scientific content, including the research question, methods, data, results, interpretations "
        f"and references, was supplied by the authors as a structured ledger of {st['claims']} claims and "
        f"{st['references']} references. The tool was restricted to expressing those claims in prose: it did not "
        f"generate ideas, select references, or assess the correctness of the claims. "
        f"Of {st['sentences']} generated sentences, {st['traced_sentences']} are linked to specific author claims and "
        f"{st['signpost_sentences']} are content-free transitions. {st['user_edited']} sentences were edited by the "
        f"authors and {st['accepted_after_review']} flagged sentences were accepted after author review; "
        f"{st['flagged']} sentences remained flagged by automated verification at the time of export. "
        f"The authors take full responsibility for the content of this manuscript."
    )


def disclosure_markdown(project: Project) -> str:
    st = stats(project)
    lines = [
        f"# AI-use disclosure: {project.ledger.title}",
        "",
        "## Statement",
        "",
        disclosure_statement(project),
        "",
        "## Measurements",
        "",
        "| Measure | Value |",
        "|---|---|",
    ]
    rows = [
        ("Tool", st["tool"]), ("Model", f"{st['model']} ({st['backend']})"),
        ("Author claims", st["claims"]), ("Claims expressed in text", st["claims_expressed"]),
        ("References (author library)", st["references"]),
        ("Sentences", st["sentences"]), ("Traceability", f"{st['traceability_pct']}%"),
        ("Sentences with verification errors", f"{st['leakage_rate_pct']}%"),
        ("Invented citations", st["invented_citations"]),
        ("Citations outside author library", st["citations_outside_library"]),
        ("Mean NLI entailment", st["mean_entailment"] if st["mean_entailment"] is not None else "n/a"),
        ("Sentences edited by authors", st["user_edited"]),
        ("Flagged sentences accepted after review", st["accepted_after_review"]),
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    if st["flag_counts"]:
        lines += ["", "## Verification flags", "", "| Flag | Count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in st["flag_counts"].items()]

    lines += ["", "## Sentence provenance", ""]
    claims = project.ledger.claim_map()
    if project.draft:
        for p in project.draft.paragraphs:
            lines.append(f"### {p.section} / {p.id}")
            lines.append("")
            for s in p.sentences:
                src = "transition (no claims)" if s.signpost else ", ".join(
                    f"{cid} ({claims[cid].type.value})" if cid in claims else cid for cid in s.claim_ids)
                mark = {"verified": "verified", "flagged": "FLAGGED", "user_edited": "edited by author",
                        "accepted": "accepted after review"}[s.status]
                lines.append(f"- **{s.id}** [{mark}] _{src}_: {s.text}")
            lines.append("")
    lines += ["## Author claim ledger", ""]
    for c in project.ledger.claims:
        refs = f" (refs: {', '.join(c.refs)})" if c.refs else ""
        lines.append(f"- **{c.id}** ({c.type.value}) {c.text}{refs}")
    lines.append("")
    return "\n".join(lines)
