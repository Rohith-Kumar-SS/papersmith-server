"""Who in your college you should talk to, and why.

A person is suggested for what they share with you (topics, methods; rarer ones count more) and above all
for filling a gap: they work on something you are looking for, or you work on something they are looking
for. Every suggestion carries its reasons, written only from what both people chose to publish.
"""

from __future__ import annotations

import math
from collections import Counter

from .. import textutils as tu
from .store import key_of

ROLES = {"ug": "UG student", "pg": "PG student", "phd": "PhD scholar", "faculty": "Faculty", "researcher": "Researcher"}
# who can mentor whom: a mentor is further along than the person asking
MENTORS_FOR = {"ug": {"pg", "phd", "faculty", "researcher"}, "pg": {"phd", "faculty", "researcher"},
               "phd": {"faculty", "researcher"}, "faculty": set(), "researcher": set(), "": {"phd", "faculty", "researcher"}}
MENTOR_WEIGHT = {"faculty": 1.5, "researcher": 1.3, "phd": 1.2, "pg": 1.0}


def visible(p: dict) -> bool:
    return bool(p.get("published")) and p.get("visibility", "community") == "community"


def public_view(p: dict, institution_name: str = "") -> dict:
    """What others may see: never papers, facts or results."""
    return {
        "uid": p["uid"], "name": p.get("name") or "Researcher", "role": p.get("role", ""),
        "role_label": ROLES.get(p.get("role", ""), ""), "department": p.get("department", ""),
        "institution": institution_name, "verified": bool(p.get("verified")), "field": p.get("field", ""),
        "bio": p.get("bio", ""), "topics": [t["name"] for t in p.get("topics", [])],
        "methods": [m["name"] for m in p.get("methods", [])], "needs": [n["text"] for n in p.get("needs", [])],
    }


_SUFFIXES = ("izations", "ization", "ations", "ation", "ating", "ated", "ates", "ings", "ing", "ions", "ion",
             "ers", "er", "ies", "es", "ed", "s")


def _root(word: str) -> str:
    """'calibrating', 'calibration', 'calibrated' -> 'calibr'; 'cities' -> 'citi'. Coarser than the checker's
    stemmer on purpose: here two people's wording only has to land on the same idea."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _stems(text: str) -> set[str]:
    return {_root(w) for w in tu.words(text.lower()) if w not in tu.STOPWORDS and len(w) > 2}


def need_matches(need: str, vocab: dict[str, str]) -> list[str]:
    """Topic / method keys a 'looking for' statement asks for: at least three quarters of the topic's words
    appear in it (so 'air quality monitoring' is not what 'air quality data from other cities' asks for)."""
    words = _stems(need)
    out = []
    for key, name in vocab.items():
        tw = _stems(name)
        if tw and len(tw & words) / len(tw) >= 0.75:
            out.append(key)
    return out


def _status(uid: str, network: dict) -> str:
    if uid in network.get("connections", []):
        return "connected"
    if any(r.get("to") == uid for r in network.get("outgoing", [])):
        return "requested"
    if any(r.get("from") == uid for r in network.get("incoming", [])):
        return "incoming"
    return "none"


def rank(me: dict, people: list[dict], network: dict, institution_name: str = "", limit: int = 12) -> dict:
    """{'collaborators': [...], 'mentors': [...]} for `me`, each {person, score, reasons, status}."""
    others = [p for p in people if p["uid"] != me["uid"] and visible(p)]
    everyone = others + [me]
    df: Counter[str] = Counter()
    vocab: dict[str, str] = {}
    for p in everyone:
        for item in p.get("topics", []) + p.get("methods", []):
            vocab[item["key"]] = item["name"]
        df.update({x["key"] for x in p.get("topics", []) + p.get("methods", [])})
    n = max(1, len(everyone))

    def idf(key: str) -> float:
        return math.log(1 + n / max(1, df[key]))

    mine_t = {t["key"]: t["name"] for t in me.get("topics", [])}
    mine_m = {m["key"]: m["name"] for m in me.get("methods", [])}
    mine = {**mine_t, **mine_m}
    my_needs = [(nd["text"], need_matches(nd["text"], vocab)) for nd in me.get("needs", [])]

    scored = []
    for q in others:
        q_t = {t["key"]: t["name"] for t in q.get("topics", [])}
        q_m = {m["key"]: m["name"] for m in q.get("methods", [])}
        theirs = {**q_t, **q_m}
        score, reasons = 0.0, []
        shared_t = [k for k in mine_t if k in q_t]
        shared_m = [k for k in mine_m if k in q_m]
        if shared_t:
            score += sum(idf(k) for k in shared_t)
            reasons.append("Both work on " + _join([q_t[k] for k in sorted(shared_t, key=idf, reverse=True)][:3]))
        if shared_m:
            score += 0.6 * sum(idf(k) for k in shared_m)
            reasons.append("Both use " + _join([q_m[k] for k in sorted(shared_m, key=idf, reverse=True)][:3]))
        for text, keys in my_needs:
            hit = [k for k in keys if k in theirs and k not in mine]      # a gap is something you don't work on yet
            if hit:
                score += 2.0 * max(idf(k) for k in hit)
                reasons.insert(0, f"Works on {theirs[hit[0]]}, which you're looking for (“{text}”)")
                break
        for nd in q.get("needs", []):
            hit = [k for k in need_matches(nd["text"], vocab) if k in mine and k not in theirs]
            if hit:
                score += 1.5 * max(idf(k) for k in hit)
                reasons.append(f"Is looking for “{nd['text']}”, and you work on {mine[hit[0]]}")
                break
        if score > 0 and me.get("field") and q.get("field") == me.get("field"):
            score += 0.3
        if score > 0:
            scored.append({"person": public_view(q, institution_name), "score": round(score, 3),
                           "reasons": reasons[:3], "status": _status(q["uid"], network)})
    scored.sort(key=lambda s: -s["score"])

    mentor_roles = MENTORS_FOR.get(me.get("role", ""), set())
    mentors = [dict(s, score=round(s["score"] * MENTOR_WEIGHT.get(s["person"]["role"], 1.0), 3))
               for s in scored if s["person"]["role"] in mentor_roles]
    mentors.sort(key=lambda s: -s["score"])
    mentor_ids = {m["person"]["uid"] for m in mentors[:limit]}
    collaborators = [s for s in scored if s["person"]["uid"] not in mentor_ids]
    return {"collaborators": collaborators[:limit], "mentors": mentors[:limit]}


def graph(people: list[dict], connections: list[tuple[str, str]], institution_name: str = "",
          max_topics: int = 80) -> dict:
    """The college's research map: people, the topics they work on, and who is connected."""
    shown = [p for p in people if visible(p)]
    count: Counter[str] = Counter()
    names: dict[str, str] = {}
    for p in shown:
        for t in p.get("topics", []):
            count[t["key"]] += 1
            names[t["key"]] = t["name"]
    topics = [k for k, _ in count.most_common(max_topics)]
    keep = set(topics)
    ids = {p["uid"] for p in shown}
    return {
        "people": [public_view(p, institution_name) for p in shown],
        "topics": [{"key": k, "name": names[k], "people": count[k]} for k in topics],
        "links": [{"person": p["uid"], "topic": t["key"]} for p in shown for t in p.get("topics", []) if t["key"] in keep],
        "connections": [{"a": a, "b": b} for a, b in connections if a in ids and b in ids],
    }


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def as_items(names: list[str], limit: int = 12) -> list[dict]:
    """Typed-in topic or method names -> stored items with canonical keys."""
    out, seen = [], set()
    for name in names:
        name = " ".join(str(name).split())[:60]
        key = key_of(name)
        if key and key not in seen:
            seen.add(key)
            out.append({"key": key, "name": name, "weight": 1.0})
    return out[:limit]
