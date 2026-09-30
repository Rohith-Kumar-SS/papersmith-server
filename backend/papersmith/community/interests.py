"""Research interests PaperSmith notices from what people do, not only from what they type into a profile.

Signals, strongest first:
  * papers they write here (title, aims, methods, contributions and what they chat about), read by the model
  * papers they published (claimed from OpenAlex): recent and well-cited ones count more
  * what they explore in the community (openings, questions, people they open), a light signal that fades fast

Each signal fades with age (a half-life per kind). What is learned shapes the person's OWN suggestions at once;
others see it (and can find them by it) only after the person confirms it, which moves it into their profile.
Private paper topics therefore never leak to anyone else.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from datetime import datetime, timezone

from .. import storage
from ..models import ClaimType, now_iso
from . import core
from .matching import as_items, tidy
from .store import key_of, store

log = logging.getLogger(__name__)

HALF_LIFE_DAYS = {"paper": 180, "chat": 90, "publication": 1460, "explore": 30}
BASE = {"paper": 1.0, "chat": 0.6, "publication": 0.5, "explore": 0.25}
NOTICED = 0.5               # weight at which an interest is shown to the person and used in their suggestions
MAX_ITEMS = 60
MAX_SOURCES = 6
PAPER_EVERY = 600           # seconds between two readings of the same paper

SYSTEM = """You name the research areas and methods a piece of work is about, so that people working on the same things can find each other.

- topics: up to 5 research areas, 1 to 4 words each, general enough to be shared by others ("cryptocurrency forensics", "air quality monitoring"); never a result or a number.
- methods: up to 5 techniques, tools or instruments the work uses ("machine learning", "graph analysis", "SEM imaging").
Name only what the notes show the work is about. Reuse a name from EXISTING when it means the same thing. Return JSON."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["topics", "methods"],
          "properties": {"topics": {"type": "array", "items": {"type": "string"}},
                         "methods": {"type": "array", "items": {"type": "string"}}}}

USEFUL = {ClaimType.objective, ClaimType.contribution, ClaimType.method, ClaimType.dataset, ClaimType.gap,
          ClaimType.background, ClaimType.future_work}


def _age_days(at: str) -> float:
    try:
        then = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400)


def _weight(item: dict) -> float:
    total = 0.0
    for src in item.get("sources", []):
        kind = src.get("type", "explore")
        total += src.get("w", BASE.get(kind, 0.25)) * 0.5 ** (_age_days(src.get("at", "")) / HALF_LIFE_DAYS.get(kind, 60))
    return min(3.0, total)


def _state(person: dict) -> dict:
    inf = person.get("inferred") or {}
    return {"items": dict(inf.get("items") or {}), "hidden": list(inf.get("hidden") or []),
            "papers": dict(inf.get("papers") or {})}


def learned(person: dict, include_confirmed: bool = False) -> list[dict]:
    """What PaperSmith noticed, strongest first: [{key, name, kind, weight, sources: [labels], confirmed}]."""
    st = _state(person)
    confirmed = {t["key"] for t in person.get("topics", [])} | {m["key"] for m in person.get("methods", [])}
    out = []
    for key, item in st["items"].items():
        if key in st["hidden"]:
            continue
        w = _weight(item)
        if w < NOTICED or (key in confirmed and not include_confirmed):
            continue
        labels = []
        for src in sorted(item.get("sources", []), key=lambda s: s.get("at", ""), reverse=True):
            label = src.get("label", "")
            if label and label not in labels:
                labels.append(label)
        out.append({"key": key, "name": item["name"], "kind": item.get("kind", "topic"), "weight": round(w, 2),
                    "sources": labels[:3], "confirmed": key in confirmed})
    return sorted(out, key=lambda x: -x["weight"])


def private_interests(person: dict) -> dict[str, dict]:
    """key -> {name, kind, weight, source} of learned, unconfirmed interests: used only for the person's own
    suggestions and never shown to anyone else."""
    return {x["key"]: {"name": x["name"], "kind": x["kind"], "weight": min(1.0, x["weight"]),
                       "source": x["sources"][0] if x["sources"] else ""} for x in learned(person)}


def record(uid: str, found: list[tuple[str, str]], source: dict) -> None:
    """Add a signal: found = [(name, 'topic'|'method')], source = {type, ref, label}."""
    s = store()
    person = s.get_person(uid)
    if not person or not found:
        return
    st = _state(person)
    src = {**source, "at": now_iso(), "w": source.get("w", BASE.get(source.get("type", ""), 0.25))}
    for name, kind in found:
        name = " ".join(name.split())[:60]
        key = key_of(name)
        if not key:
            continue
        item = st["items"].setdefault(key, {"name": name, "kind": kind, "sources": []})
        # one entry per (type, ref): reading the same paper again refreshes it instead of stacking
        item["sources"] = [x for x in item["sources"] if (x.get("type"), x.get("ref")) != (src.get("type"), src.get("ref"))]
        item["sources"].append(src)
        item["sources"] = sorted(item["sources"], key=lambda x: x.get("at", ""))[-MAX_SOURCES:]
    if len(st["items"]) > MAX_ITEMS:
        keep = sorted(st["items"].items(), key=lambda kv: -_weight(kv[1]))[:MAX_ITEMS]
        st["items"] = dict(keep)
    s.upsert_person(uid, inferred=st)


def confirm(uid: str, key: str) -> dict:
    """Move a learned interest into the public profile."""
    s = store()
    person = s.get_person(uid) or {}
    item = _state(person)["items"].get(key)
    if not item:
        raise ValueError("That interest isn't there any more.")
    field = "methods" if item.get("kind") == "method" else "topics"
    names = [x["name"] for x in person.get(field, [])]
    if key not in {x["key"] for x in person.get(field, [])}:
        if len(names) >= 12:
            raise ValueError(f"Your profile already has 12 {field}. Remove one first.")
        names.append(item["name"])
    return s.upsert_person(uid, **{field: as_items(names, 12)})


def hide(uid: str, key: str) -> dict:
    s = store()
    st = _state(s.get_person(uid) or {})
    if key not in st["hidden"]:
        st["hidden"].append(key)
    return s.upsert_person(uid, inferred=st)


# ---------------------------------------------------------------- from papers written here

def paper_notes(pid: str) -> tuple[str, str, list[str]]:
    """(title, notes for the model, the people whose work it is)."""
    p = storage.load(pid)
    title = p.ledger.title or p.name
    lines = [f"TITLE: {title}"]
    for c in p.ledger.claims:
        if c.type in USEFUL and not c.refs:
            lines.append(f"- {c.text[:180]}")
    said = [m.text[:200] for m in p.messages if m.role == "user" and not m.data.get("event") and len(m.text) > 20][-8:]
    if said:
        lines.append("WHAT THE RESEARCHER SAID:")
        lines += [f"- {t}" for t in said]
    people = [p.owner] + [m.uid for m in p.members if m.role == "author"]
    return title, "\n".join(lines)[:5000], [u for u in people if u]


_timers: dict[str, threading.Timer] = {}
_timer_lock = threading.Lock()
_last_read: dict[str, float] = {}


def touch(pid: str, delay: float = 90.0) -> None:
    """Something changed in a paper: read it again a little later (many changes, one reading)."""
    if not core.settings.hosted:
        return
    with _timer_lock:
        if pid in _timers:
            return
        t = threading.Timer(delay, _run, args=(pid,))
        t.daemon = True
        _timers[pid] = t
        t.start()


def _run(pid: str) -> None:
    with _timer_lock:
        _timers.pop(pid, None)
    if time.time() - _last_read.get(pid, 0) < PAPER_EVERY:
        touch(pid, PAPER_EVERY)
        return
    try:
        learn_from_paper(pid)
    except Exception:  # noqa: BLE001 - a bonus; never breaks writing
        log.exception("reading paper %s for interests failed", pid)


def learn_from_paper(pid: str) -> dict | None:
    try:
        title, notes, people = paper_notes(pid)
    except (KeyError, ValueError):
        return None
    if len(notes) < 60 or not people:
        return None
    _last_read[pid] = time.time()
    fingerprint = hashlib.sha1(notes.encode()).hexdigest()[:16]
    s = store()
    owner = s.get_person(people[0])
    if not owner:
        return None
    previous = s.get_doc("paper_topics", pid)
    if previous and previous.get("fingerprint") == fingerprint:
        return previous
    vocab = set()
    if owner.get("institution"):
        for m in s.members(owner["institution"])[:300]:
            vocab.update(t["name"] for t in m.get("topics", []) + m.get("methods", []))
    prompt = f"EXISTING: {', '.join(sorted(vocab)[:120]) or '(none yet)'}\n\nNOTES:\n{notes}"
    raw = core.ask(SYSTEM, prompt, SCHEMA, max_tokens=800)
    topics = as_items(tidy([str(t) for t in raw.get("topics", []) if str(t).strip()]), 5)
    methods = as_items(tidy([str(m) for m in raw.get("methods", []) if str(m).strip()]), 5)
    if not topics and not methods:
        return previous
    doc = {"id": pid, "owner": people[0], "people": people, "institution": owner.get("institution", ""),
           "title": title, "topics": topics, "methods": methods, "fingerprint": fingerprint, "at": now_iso()}
    s.put_doc("paper_topics", doc)
    found = [(t["name"], "topic") for t in topics] + [(m["name"], "method") for m in methods]
    for uid in people:
        record(uid, found, {"type": "paper", "ref": pid, "label": f"your paper “{title[:60]}”"})
    from . import related
    related.announce(doc)
    return doc


# ---------------------------------------------------------------- from exploring the community

def explored(uid: str, items: list[dict], label: str, ref: str) -> None:
    """They opened an opening, question or profile about these topics: a light, fast-fading signal."""
    found = [(x["name"], x.get("kind", "topic")) for x in items if x.get("name")][:6]
    if found:
        core.in_background(record, uid, found, {"type": "explore", "ref": ref, "label": label})


# ---------------------------------------------------------------- from published papers

def learn_from_works(uid: str, works: list[dict]) -> None:
    """Topics of the person's published papers: recent and cited ones weigh more."""
    per_topic: dict[str, list[dict]] = {}
    for w in works:
        year = w.get("year") or 0
        recency = 0.5 ** (max(0, time.gmtime().tm_year - year) / 4) if year else 0.3
        weight = BASE["publication"] * recency * (1 + math.log1p(w.get("cited_by", 0)) / 6)
        for t in w.get("topics", [])[:3]:
            per_topic.setdefault(t, []).append({"w": weight, "title": w.get("title", "")})
    s = store()
    person = s.get_person(uid)
    if not person:
        return
    st = _state(person)
    for name, hits in per_topic.items():
        key = key_of(name)
        if not key:
            continue
        item = st["items"].setdefault(key, {"name": name, "kind": "topic", "sources": []})
        item["sources"] = [x for x in item["sources"] if x.get("type") != "publication"]
        n = len(hits)
        item["sources"].append({"type": "publication", "ref": "scholar", "at": now_iso(), "w": min(2.0, sum(h["w"] for h in hits)),
                                "label": f"{n} of your published papers" if n > 1 else f"your paper “{hits[0]['title'][:60]}”"})
    if len(st["items"]) > MAX_ITEMS:
        st["items"] = dict(sorted(st["items"].items(), key=lambda kv: -_weight(kv[1]))[:MAX_ITEMS])
    s.upsert_person(uid, inferred=st)
