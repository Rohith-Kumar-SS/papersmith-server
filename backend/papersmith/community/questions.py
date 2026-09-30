"""/api/community/questions: ask your college a research question.

The question's topics are worked out (from the college's own vocabulary, or by the model), and it goes to the
few people most likely to know: their profiles and published papers cover those topics. Answers that the
asker accepts or others mark helpful are credited on the answerer's profile ("helped 4 people with questions
on machine learning"), which also counts as evidence when they apply to an opening.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models import now_iso
from . import core, interests, matching
from .store import store

router = APIRouter(prefix="/api/community/questions")

TOPIC_SYSTEM = """Name what a research question is about, so it can be sent to people who know the area.
- topics: 1 to 3 research areas, 1 to 4 words each.
- methods: up to 3 techniques or tools the question is about.
Reuse a name from EXISTING when it means the same thing. Return JSON."""

TOPIC_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["topics", "methods"],
                "properties": {"topics": {"type": "array", "items": {"type": "string"}},
                               "methods": {"type": "array", "items": {"type": "string"}}}}


def _view(q: dict, me: dict, full: bool = False) -> dict:
    s = store()
    out = {k: q.get(k) for k in ("id", "title", "created_at", "status", "accepted")}
    out.update({"topics": [t["name"] for t in q.get("topics", []) + q.get("methods", [])],
                "owner": {"uid": q["owner"], "name": core.display_name(q["owner"])}, "mine": q["owner"] == me["uid"],
                "answers_count": len(q.get("answers", [])), "for_me": me["uid"] in q.get("routed", [])})
    if full:
        out["body"] = q.get("body", "")
        out["answers"] = [{"id": a["id"], "body": a["body"], "at": a["at"], "helpful": len(a.get("helpful", [])),
                           "marked": me["uid"] in a.get("helpful", []), "mine": a["owner"] == me["uid"],
                           "author": _author(s, a["owner"])} for a in q.get("answers", [])]
        out["answers"].sort(key=lambda a: (a["id"] != q.get("accepted"), -a["helpful"], a["at"]))
    else:
        out["excerpt"] = q.get("body", "")[:180]
    return out


def _author(s, uid: str) -> dict:
    p = s.get_person(uid) or {"uid": uid}
    return {"uid": uid, "name": p.get("name") or "Researcher", "role_label": core.ROLES.get(p.get("role", ""), ""),
            "department": p.get("department", "")}


def _load(qid: str, me: dict) -> dict:
    q = store().get_doc("question", qid)
    core.same_college(me, q)
    return q  # type: ignore[return-value]


@router.get("")
def list_questions(filter: str = ""):
    me = core.ensure_person(core.me_uid())
    if not me.get("institution"):
        return []
    qs = store().list_docs("question", institution=me["institution"], limit=300)
    if filter == "mine":
        qs = [q for q in qs if q["owner"] == me["uid"]]
    elif filter == "for_me":
        qs = [q for q in qs if me["uid"] in q.get("routed", []) and q.get("status") == "open"]
    elif filter == "open":
        qs = [q for q in qs if q.get("status") == "open"]
    return [_view(q, me) for q in qs]


@router.get("/{qid}")
def get_question(qid: str):
    me = core.ensure_person(core.me_uid())
    q = _load(qid, me)
    if q["owner"] != me["uid"]:
        interests.explored(me["uid"], q.get("topics", []) + q.get("methods", []), "questions you read", f"question:{qid}")
    return _view(q, me, full=True)


class QuestionIn(BaseModel):
    title: str
    body: str = ""


def _topics(text: str, institution: str) -> tuple[list[dict], list[dict]]:
    s = store()
    vocab_t: dict[str, str] = {}
    vocab_m: dict[str, str] = {}
    for p in s.members(institution):
        vocab_t.update({t["key"]: t["name"] for t in p.get("topics", [])})
        vocab_m.update({m["key"]: m["name"] for m in p.get("methods", [])})
    found_t = matching.need_matches(text, vocab_t)
    found_m = matching.need_matches(text, vocab_m)
    if found_t or found_m:
        return (matching.as_items([vocab_t[k] for k in found_t], 3), matching.as_items([vocab_m[k] for k in found_m], 3))
    try:
        raw = core.ask(TOPIC_SYSTEM, f"EXISTING: {', '.join(sorted(set(vocab_t.values()) | set(vocab_m.values()))[:120]) or '(none)'}\n\n"
                                     f"QUESTION: {text[:1500]}", TOPIC_SCHEMA, max_tokens=300)
    except ValueError:
        return [], []
    return (matching.as_items(matching.tidy(raw.get("topics", [])), 3),
            matching.as_items(matching.tidy(raw.get("methods", [])), 3))


@router.post("")
def ask(body: QuestionIn):
    me = core.ensure_person(core.me_uid())
    if not me.get("institution"):
        raise HTTPException(400, "Join your college first (Your profile).")
    title = " ".join(body.title.split())[:200]
    if len(title) < 10:
        raise HTTPException(400, "Write the question in a sentence.")
    s = store()
    recent = [q for q in s.list_docs("question", owner=me["uid"], limit=20) if q["created_at"][:10] == now_iso()[:10]]
    if len(recent) >= 5:
        raise HTTPException(429, "You can ask 5 questions a day.")
    topics, methods = _topics(f"{title}. {body.body}", me["institution"])
    q = {"id": core.new_id("q"), "owner": me["uid"], "institution": me["institution"], "title": title,
         "body": body.body.strip()[:4000], "topics": topics, "methods": methods, "answers": [], "accepted": "",
         "status": "open", "created_at": now_iso()}
    routed = matching.route_question(q, s.members(me["institution"]))
    q["routed"] = [p["uid"] for p, _ in routed]
    s.put_doc("question", q)
    for p, names in routed:
        core.notify(p["uid"], "question", f"A question you could answer (you work on {matching._join(names[:2])}): “{title[:90]}”",
                    link=f"/community/ask/{q['id']}", actor=me["uid"])
    return dict(_view(q, me, full=True), routed_to=len(routed))


class AnswerIn(BaseModel):
    body: str


@router.post("/{qid}/answers")
def answer(qid: str, body: AnswerIn):
    me = core.ensure_person(core.me_uid())
    q = _load(qid, me)
    text = body.body.strip()[:4000]
    if len(text) < 5:
        raise HTTPException(400, "Write an answer first.")
    q.setdefault("answers", []).append({"id": core.new_id("a"), "owner": me["uid"], "body": text, "at": now_iso(), "helpful": []})
    store().put_doc("question", q)
    core.notify(q["owner"], "answer", f"{me.get('name', 'Someone')} answered your question “{q['title'][:80]}”",
                link=f"/community/ask/{qid}", actor=me["uid"])
    return _view(q, me, full=True)


@router.post("/{qid}/answers/{aid}/helpful")
def helpful(qid: str, aid: str):
    me = core.ensure_person(core.me_uid())
    q = _load(qid, me)
    a = next((x for x in q.get("answers", []) if x["id"] == aid), None)
    if not a:
        raise HTTPException(404, "answer not found")
    if a["owner"] == me["uid"]:
        raise HTTPException(400, "You can't mark your own answer.")
    marks = a.setdefault("helpful", [])
    if me["uid"] in marks:
        marks.remove(me["uid"])
    else:
        marks.append(me["uid"])
    store().put_doc("question", q)
    return _view(q, me, full=True)


@router.post("/{qid}/answers/{aid}/accept")
def accept(qid: str, aid: str):
    me = core.ensure_person(core.me_uid())
    q = _load(qid, me)
    if q["owner"] != me["uid"]:
        raise HTTPException(403, "Only the person who asked can accept an answer.")
    a = next((x for x in q.get("answers", []) if x["id"] == aid), None)
    if not a:
        raise HTTPException(404, "answer not found")
    q["accepted"] = "" if q.get("accepted") == aid else aid
    q["status"] = "answered" if q["accepted"] else "open"
    store().put_doc("question", q)
    if q["accepted"]:
        core.notify(a["owner"], "accepted_answer", f"{me.get('name', 'The asker')} accepted your answer to “{q['title'][:80]}”",
                    link=f"/community/ask/{qid}", actor=me["uid"])
    return _view(q, me, full=True)


@router.delete("/{qid}")
def delete(qid: str):
    me = core.ensure_person(core.me_uid())
    q = _load(qid, me)
    if q["owner"] != me["uid"] and not core.is_admin(me["uid"], me["institution"]):
        raise HTTPException(403, "Only the person who asked (or a college admin) can remove it.")
    store().delete_doc("question", qid)
    return {"ok": True}


def credits(uid: str, institution: str) -> dict:
    """{'helped': people helped, 'topics': [names]} for a profile."""
    helped, topics = 0, {}
    for q in store().list_docs("question", institution=institution, limit=500):
        for a in q.get("answers", []):
            if a["owner"] == uid and (a.get("helpful") or q.get("accepted") == a["id"]):
                helped += 1
                for t in q.get("topics", []):
                    topics[t["name"]] = topics.get(t["name"], 0) + 1
    return {"helped": helped, "topics": [t for t, _ in sorted(topics.items(), key=lambda kv: -kv[1])[:4]]}
