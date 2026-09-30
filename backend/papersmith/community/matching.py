"""Who in your college you should talk to, which openings fit you, who should answer a question, and why.

A person is suggested for what they share with you (topics, methods; rarer ones count more) and above all
for filling a gap: they work on something you are looking for, or you work on something they are looking
for. Every suggestion carries its reasons, written only from what the other person chose to publish; your
own learned interests (from your papers, chats and exploring) are used for YOUR suggestions only.
"""

from __future__ import annotations

import math
from collections import Counter

from .. import textutils as tu
from .core import MENTOR_ROLES, ROLES
from .store import key_of

# who can mentor whom: a mentor is further along than the person asking
MENTORS_FOR = {"ug": {"pg", "phd", "faculty", "researcher"}, "pg": {"phd", "faculty", "researcher"},
               "phd": {"faculty", "researcher"}, "faculty": set(), "researcher": set(), "staff": set(),
               "": {"phd", "faculty", "researcher"}}
MENTOR_WEIGHT = {"faculty": 1.5, "researcher": 1.3, "phd": 1.2, "pg": 1.0}


def visible(p: dict) -> bool:
    return bool(p.get("published")) and p.get("visibility", "community") == "community"


def public_view(p: dict, institution_name: str = "") -> dict:
    """What others may see: never papers, facts, results or anything PaperSmith learned but they didn't confirm."""
    scholar = p.get("scholar") or {}
    return {
        "uid": p["uid"], "name": p.get("name") or "Researcher", "role": p.get("role", ""),
        "role_label": ROLES.get(p.get("role", ""), ""), "department": p.get("department", ""),
        "institution": institution_name, "verified": bool(p.get("verified")), "role_verified": bool(p.get("role_verified")),
        "field": p.get("field", ""), "bio": p.get("bio", ""), "topics": [t["name"] for t in p.get("topics", [])],
        "methods": [m["name"] for m in p.get("methods", [])], "needs": [n["text"] for n in p.get("needs", [])],
        "publications": {"count": scholar.get("works_count", 0), "recent": scholar.get("recent", [])[:3],
                         "source": scholar.get("source", "")} if scholar.get("id") else None,
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


def related_keys(a: str, b: str) -> bool:
    """Two topic names about the same thing: the same key, or one's words inside the other's
    ('machine learning' / 'machine learning for forensics')."""
    if a == b:
        return True
    wa, wb = _stems(a), _stems(b)
    small, big = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    return bool(small) and len(small) >= 1 and small <= big and len(small) / len(big) >= 0.5


def _status(uid: str, network: dict) -> str:
    if uid in network.get("connections", []):
        return "connected"
    if any(r.get("to") == uid for r in network.get("outgoing", [])):
        return "requested"
    if any(r.get("from") == uid for r in network.get("incoming", [])):
        return "incoming"
    return "none"


def _idf_over(items_per_doc: list[set[str]]):
    df: Counter[str] = Counter()
    for keys in items_per_doc:
        df.update(keys)
    n = max(1, len(items_per_doc))
    return lambda key: math.log(1 + n / max(1, df[key]))


def rank(me: dict, people: list[dict], network: dict, institution_name: str = "", limit: int = 12,
         private: dict[str, dict] | None = None) -> dict:
    """{'collaborators': [...], 'mentors': [...]} for `me`, each {person, score, reasons, status}.
    `private`: me's learned, unconfirmed interests (key -> {name, kind, weight, source}), used only here."""
    private = private or {}
    others = [p for p in people if p["uid"] != me["uid"] and visible(p)]
    everyone = others + [me]
    vocab: dict[str, str] = {}
    for p in everyone:
        for item in p.get("topics", []) + p.get("methods", []):
            vocab[item["key"]] = item["name"]
    idf = _idf_over([{x["key"] for x in p.get("topics", []) + p.get("methods", [])} for p in everyone])

    mine_t = {t["key"]: t["name"] for t in me.get("topics", [])}
    mine_m = {m["key"]: m["name"] for m in me.get("methods", [])}
    learned_t = {k: v for k, v in private.items() if v["kind"] != "method" and k not in mine_t}
    learned_m = {k: v for k, v in private.items() if v["kind"] == "method" and k not in mine_m}
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
        # what PaperSmith noticed about me (only I see these reasons)
        # (across kinds: what I learned as a method may be someone's topic, e.g. deep learning)
        hit_t = [k for k in learned_t if k in theirs]
        hit_m = [k for k in learned_m if k in theirs]
        if hit_t or hit_m:
            best = max(hit_t + hit_m, key=lambda k: idf(k) * private[k]["weight"])
            score += 0.8 * sum(idf(k) * private[k]["weight"] for k in hit_t) + 0.5 * sum(idf(k) * private[k]["weight"] for k in hit_m)
            src = private[best]["source"]
            reasons.append(f"Works on {theirs[best]}, which comes up in {src}" if src else f"Works on {theirs[best]}, which you've been exploring")
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


def interest_weights(person: dict, private: dict[str, dict] | None = None) -> dict[str, tuple[str, float, str]]:
    """key -> (name, weight, why) for everything a person is interested in: their profile (weight 1) and, when
    given, what PaperSmith learned (their own view only)."""
    out: dict[str, tuple[str, float, str]] = {}
    for t in person.get("topics", []):
        out[t["key"]] = (t["name"], 1.0, "profile")
    for m in person.get("methods", []):
        out[m["key"]] = (m["name"], 0.8, "profile")
    for k, v in (private or {}).items():
        if k not in out:
            out[k] = (v["name"], 0.8 * v["weight"], v["source"] or "what you've been exploring")
    return out


def _item_hits(wants: dict[str, tuple[str, float, str]], items: list[dict]) -> list[tuple[str, str, float, str]]:
    """(item name, matched interest, weight, why) for items (topics/methods of an opening or question) that a
    person's interests cover, by key or by related wording."""
    hits = []
    for it in items:
        best = None
        for k, (name, w, why) in wants.items():
            if k == it["key"] or related_keys(k, it["key"]):
                if best is None or w > best[2]:
                    best = (it["name"], name, w, why)
        if best:
            hits.append(best)
    return hits


def rank_openings(me: dict, openings: list[dict], private: dict[str, dict] | None = None, limit: int = 20) -> list[dict]:
    """Openings (a professor's problem statement, or a student looking for a mentor) that fit me, with reasons."""
    wants = interest_weights(me, private)
    idf = _idf_over([{x["key"] for x in o.get("topics", []) + o.get("methods", [])} for o in openings])
    out = []
    for o in openings:
        if o.get("owner") == me["uid"] or o.get("status", "open") != "open":
            continue
        hits = _item_hits(wants, o.get("topics", []) + o.get("methods", []))
        if not hits:
            continue
        score = sum(w * idf(key_of(name)) for name, _, w, _ in hits)
        profile = [h for h in hits if h[3] == "profile"]
        noticed = [h for h in hits if h[3] != "profile"]
        reasons = []
        if profile:
            reasons.append("Matches your work on " + _join(sorted({h[1] for h in profile})[:3]))
        if noticed:
            reasons.append(f"Needs {noticed[0][0]}, which comes up in {noticed[0][3]}")
        if o.get("type") == "seeking" and me.get("role") in MENTOR_ROLES:
            score *= 1.2
        out.append({"opening": o, "score": round(score, 3), "reasons": reasons[:2]})
    out.sort(key=lambda x: -x["score"])
    return out[:limit]


def fit(person: dict, opening: dict, credits: dict | None = None) -> dict:
    """How well an applicant fits an opening, from their public profile and published papers only:
    {score, evidence: [str], missing: [names]}."""
    items = opening.get("topics", []) + opening.get("methods", [])
    wants = interest_weights(person)
    hits = _item_hits(wants, items)
    covered = {h[0] for h in hits}
    evidence = [f"Profile lists {h[1]}" + (f" (for {h[0]})" if key_of(h[1]) != key_of(h[0]) else "") for h in hits]
    for w in (person.get("scholar") or {}).get("recent", []):
        text = f"{w.get('title', '')} {' '.join(w.get('topics', []))}"
        about = [it["name"] for it in items if _stems(it["name"]) and _stems(it["name"]) <= _stems(text)]
        if about:
            evidence.append(f"Published “{w.get('title', '')[:90]}” ({w.get('year', '')}), on {_join(about[:2])}")
            covered.update(about)
    for topic, n in (credits or {}).items():
        if any(related_keys(key_of(topic), it["key"]) for it in items):
            evidence.append(f"Helped {n} {'person' if n == 1 else 'people'} with questions on {topic}")
    missing = [it["name"] for it in items if it["name"] not in covered]
    score = len(covered) / max(1, len(items)) + 0.1 * min(3, len(evidence))
    if person.get("verified"):
        evidence.append("Verified college email")
    return {"score": round(score, 3), "evidence": evidence[:6], "missing": missing[:6], "covered": sorted(covered)}


def route_question(question: dict, people: list[dict], limit: int = 5) -> list[tuple[dict, list[str]]]:
    """The people most likely to know the answer (their profile and published papers), with what matched."""
    items = question.get("topics", []) + question.get("methods", [])
    out = []
    for p in people:
        if p["uid"] == question.get("owner") or not visible(p):
            continue
        hits = _item_hits(interest_weights(p), items)
        if hits:
            bonus = MENTOR_WEIGHT.get(p.get("role", ""), 1.0)
            out.append((sum(h[2] for h in hits) * bonus, p, [h[0] for h in hits]))
    out.sort(key=lambda x: -x[0])
    return [(p, names) for _, p, names in out[:limit]]


def graph(people: list[dict], connections: list[tuple[str, str]], institution_name: str = "",
          max_topics: int = 80, coauthors: list[tuple[str, str]] | None = None, docs: list[dict] | None = None) -> dict:
    """A research map: people, the topics they work on, who is connected, who wrote together, and (for college
    admins) the openings and projects that sit on those topics."""
    shown = [p for p in people if visible(p)]
    count: Counter[str] = Counter()
    names: dict[str, str] = {}
    for p in shown:
        for t in p.get("topics", []):
            count[t["key"]] += 1
            names[t["key"]] = t["name"]
    for d in docs or []:
        for t in d.get("topics", []):
            count[t["key"]] += 0
            names.setdefault(t["key"], t["name"])
    topics = [k for k, _ in count.most_common(max_topics)]
    keep = set(topics)
    ids = {p["uid"] for p in shown}
    return {
        "people": [public_view(p, institution_name) for p in shown],
        "topics": [{"key": k, "name": names[k], "people": count[k]} for k in topics],
        "links": [{"person": p["uid"], "topic": t["key"]} for p in shown for t in p.get("topics", []) if t["key"] in keep],
        "connections": [{"a": a, "b": b} for a, b in connections if a in ids and b in ids],
        "coauthors": [{"a": a, "b": b} for a, b in coauthors or [] if a in ids and b in ids],
        "docs": [{"id": d["id"], "kind": d["kind"], "label": d.get("title", "")[:60], "owner": d.get("owner", ""),
                  "topics": [t["key"] for t in d.get("topics", []) if t["key"] in keep]} for d in docs or []],
    }


def _join(items: list[str]) -> str:
    items = list(items)
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def tidy(names: list) -> list[str]:
    """Model-written names in sentence case: 'Air Quality Monitoring' -> 'Air quality monitoring', while keeping
    acronyms and brand spellings ('IoT', 'LoRa', 'PM2.5', 'SEM imaging')."""
    out = []
    for name in names:
        words = " ".join(str(name).split()).split(" ")
        fixed = [w if i == 0 or not (w[:1].isupper() and w[1:].islower()) else w.lower() for i, w in enumerate(words)]
        if fixed and fixed[0][:1].islower():
            fixed[0] = fixed[0][:1].upper() + fixed[0][1:]
        out.append(" ".join(fixed))
    return out


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
