"""A researcher's community profile, drafted from their own papers and approved by them before anyone sees it.

The draft names research areas, methods and what the person could use help with, in general terms: it
never repeats results, numbers or anything else that would give away unpublished work.
"""

from __future__ import annotations

import logging

from .. import storage
from ..llm import BackendError, get_backend
from ..models import ClaimType
from .matching import as_items
from .store import store

log = logging.getLogger(__name__)

FIELDS = ["Computer Science", "Electrical & Electronics", "Electronics & Communication", "Mechanical Engineering",
          "Civil Engineering", "Chemical Engineering", "Biotechnology", "Biomedical Engineering",
          "Environmental Science", "Physics", "Chemistry", "Mathematics", "Medicine & Health", "Agriculture",
          "Materials Science", "Management & Economics", "Social Sciences", "Humanities", "Design & Architecture", "Other"]

SYSTEM = """You write a short public profile for a researcher from notes about their own work, so that people in their college with related interests can find them.

- topics: 2 to 6 research areas, 1 to 4 words each, general enough to be shared ("air quality monitoring", "sensor calibration"), never a finding or a number.
- methods: up to 6 techniques, tools or instruments they use ("gradient boosting", "SEM imaging", "survey design").
- needs: up to 3 things they could use help with, drawn from their limitations, gaps and future work, phrased as a short request ("air quality data from other cities", "help with deep learning on edge devices"). Never reveal results, numbers or unpublished details.
- bio: one plain sentence about what they work on, no results.
- field: the closest field from the list.
Reuse a name from EXISTING TOPICS / EXISTING METHODS when it means the same thing. Return JSON."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["field", "topics", "methods", "needs", "bio"],
          "properties": {"field": {"type": "string", "enum": FIELDS},
                         "topics": {"type": "array", "items": {"type": "string"}},
                         "methods": {"type": "array", "items": {"type": "string"}},
                         "needs": {"type": "array", "items": {"type": "string"}},
                         "bio": {"type": "string"}}}

USEFUL = {ClaimType.objective: "aim", ClaimType.contribution: "contribution", ClaimType.method: "method",
          ClaimType.dataset: "data", ClaimType.gap: "gap", ClaimType.limitation: "limitation",
          ClaimType.future_work: "future work"}


def notes_for(uid: str, chars: int = 6000) -> str:
    """What the person's own papers say about their work (only for the drafting model, never shown to others)."""
    lines = []
    for summary in storage.list_all(owner=uid):
        try:
            p = storage.load(summary["id"])
        except (KeyError, ValueError):
            continue
        if p.owner and p.owner != uid:
            continue                        # co-authored papers belong to their owner's profile
        lines.append(f"PAPER: {p.ledger.title or p.name}")
        for c in p.ledger.claims:
            if c.type in USEFUL and not c.refs:
                lines.append(f"- ({USEFUL[c.type]}) {c.text[:200]}")
    return "\n".join(lines)[:chars]


def draft(uid: str) -> dict:
    s = store()
    me = s.get_person(uid) or {}
    notes = notes_for(uid)
    if not notes.strip():
        raise ValueError("Write or start a paper first, or type your topics yourself.")
    vocab_t, vocab_m = set(), set()
    if me.get("institution"):
        for p in s.members(me["institution"])[:300]:
            vocab_t.update(t["name"] for t in p.get("topics", []))
            vocab_m.update(m["name"] for m in p.get("methods", []))
    prompt = (f"EXISTING TOPICS: {', '.join(sorted(vocab_t)[:80]) or '(none yet)'}\n"
              f"EXISTING METHODS: {', '.join(sorted(vocab_m)[:80]) or '(none yet)'}\n\nNOTES:\n{notes}")
    try:
        raw = get_backend(role="read").generate_json(SYSTEM, prompt, SCHEMA, max_tokens=1500)
    except BackendError as exc:
        log.info("profile draft failed: %s", exc)
        raise ValueError(f"Couldn't draft your profile right now ({exc}). Try again in a minute.") from exc
    return {
        "field": raw.get("field") if raw.get("field") in FIELDS else "Other",
        "topics": [t["name"] for t in as_items(raw.get("topics", []), 6)],
        "methods": [m["name"] for m in as_items(raw.get("methods", []), 6)],
        "needs": [" ".join(str(n).split())[:120] for n in raw.get("needs", []) if str(n).strip()][:3],
        "bio": " ".join(str(raw.get("bio", "")).split())[:240],
    }
