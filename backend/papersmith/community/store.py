"""Where the community lives. Neo4j when NEO4J_URI is set (the hosted version); otherwise an in-memory
graph kept in a JSON file (tests and the laptop). Both expose the same small interface, and matching,
mentors and the explorer only use that interface.

Graph model (Neo4j):
  (:Person {uid, name, role, department, verified, visibility, published, field, bio, needs, email_domain})
  (:Person)-[:MEMBER_OF]->(:Institution {key, name, domains})
  (:Person)-[:WORKS_ON {weight}]->(:Topic {key, name})
  (:Person)-[:USES {weight}]->(:Method {key, name})
  (:Person)-[:REQUESTED {message, at}]->(:Person)      a pending connection request
  (:Person)-[:CONNECTED {since}]-(:Person)             an accepted connection
  (:Person)-[:COAUTHOR {pid}]-(:Person)                 people who wrote a paper together
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from ..config import settings
from ..models import now_iso

PERSON_FIELDS = ("uid", "name", "role", "department", "verified", "visibility", "published", "field", "bio",
                 "email_domain", "institution", "needs", "topics", "methods", "updated_at", "joined_at")


def key_of(text: str) -> str:
    """Canonical key for a topic, method or college name: 'Sensor Calibrations' -> 'sensor calibration'."""
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    out = []
    for w in words:
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return " ".join(out)[:80]


def _blank_person(uid: str) -> dict:
    return {"uid": uid, "name": "", "role": "", "department": "", "verified": False, "visibility": "community",
            "published": False, "field": "", "bio": "", "email_domain": "", "institution": "", "needs": [],
            "topics": [], "methods": [], "updated_at": now_iso(), "joined_at": now_iso()}


# ================================================================ in-memory graph (tests, laptop)

class MemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.people: dict[str, dict] = {}
        self.institutions: dict[str, dict] = {}
        self.requests: dict[tuple[str, str], dict] = {}       # (from, to) -> {message, at}
        self.connections: dict[frozenset, str] = {}          # {a, b} -> since
        self.coauthors: set[frozenset] = set()
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.people = data.get("people", {})
            self.institutions = data.get("institutions", {})
            self.requests = {(r["from"], r["to"]): {"message": r["message"], "at": r["at"]} for r in data.get("requests", [])}
            self.connections = {frozenset(c["pair"]): c["since"] for c in data.get("connections", [])}
            self.coauthors = {frozenset(c) for c in data.get("coauthors", [])}

    def _save(self) -> None:
        if not self.path:
            return
        data = {"people": self.people, "institutions": self.institutions,
                "requests": [{"from": a, "to": b, **r} for (a, b), r in self.requests.items()],
                "connections": [{"pair": sorted(k), "since": v} for k, v in self.connections.items()],
                "coauthors": [sorted(c) for c in self.coauthors]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------------- colleges
    def institution(self, key: str) -> dict | None:
        return self.institutions.get(key)

    def upsert_institution(self, name: str, domain: str = "") -> dict:
        with self._lock:
            key = key_of(name)
            inst = self.institutions.setdefault(key, {"key": key, "name": name.strip(), "domains": []})
            if domain and domain not in inst["domains"]:
                inst["domains"].append(domain)
            self._save()
            return dict(inst)

    def institution_for_domain(self, domain: str) -> dict | None:
        for inst in self.institutions.values():
            if any(domain == d or domain.endswith("." + d) for d in inst["domains"]):
                return dict(inst)
        return None

    def search_institutions(self, q: str, limit: int = 10) -> list[dict]:
        qk = key_of(q)
        counts: dict[str, int] = {}
        for p in self.people.values():
            counts[p.get("institution", "")] = counts.get(p.get("institution", ""), 0) + 1
        found = [dict(i, members=counts.get(i["key"], 0)) for i in self.institutions.values()
                 if not qk or qk in i["key"] or any(qk in d for d in i["domains"])]
        return sorted(found, key=lambda i: (-i["members"], i["name"]))[:limit]

    # ---------------------------------------------------------------- people
    def get_person(self, uid: str) -> dict | None:
        p = self.people.get(uid)
        return json.loads(json.dumps(p)) if p else None

    def upsert_person(self, uid: str, **fields) -> dict:
        with self._lock:
            p = self.people.setdefault(uid, _blank_person(uid))
            for k, v in fields.items():
                if k in PERSON_FIELDS and k != "uid":
                    p[k] = v
            p["updated_at"] = now_iso()
            self._save()
            return json.loads(json.dumps(p))

    def members(self, institution: str) -> list[dict]:
        return [json.loads(json.dumps(p)) for p in self.people.values() if p.get("institution") == institution]

    # ---------------------------------------------------------------- connections
    def request(self, a: str, b: str, message: str) -> None:
        with self._lock:
            if frozenset((a, b)) in self.connections:
                return
            if (b, a) in self.requests:              # they asked first: this is an acceptance
                self.accept(b, a)
                return
            self.requests[(a, b)] = {"message": message[:500], "at": now_iso()}
            self._save()

    def accept(self, a: str, b: str) -> bool:
        """b accepts a's request."""
        with self._lock:
            if self.requests.pop((a, b), None) is None:
                return False
            self.connections[frozenset((a, b))] = now_iso()
            self._save()
            return True

    def decline(self, a: str, b: str) -> None:
        with self._lock:
            self.requests.pop((a, b), None)
            self._save()

    def disconnect(self, a: str, b: str) -> None:
        with self._lock:
            self.connections.pop(frozenset((a, b)), None)
            self.requests.pop((a, b), None)
            self.requests.pop((b, a), None)
            self._save()

    def network(self, uid: str) -> dict:
        return {
            "connections": sorted(next(iter(k - {uid})) for k in self.connections if uid in k),
            "incoming": [{"from": a, **r} for (a, b), r in self.requests.items() if b == uid],
            "outgoing": [{"to": b, **r} for (a, b), r in self.requests.items() if a == uid],
            "coauthors": sorted(next(iter(k - {uid})) for k in self.coauthors if uid in k),
        }

    def add_coauthors(self, uids: list[str]) -> None:
        with self._lock:
            for i, a in enumerate(uids):
                for b in uids[i + 1:]:
                    if a != b:
                        self.coauthors.add(frozenset((a, b)))
            self._save()

    def connections_among(self, institution: str) -> list[tuple[str, str]]:
        members = {uid for uid, p in self.people.items() if p.get("institution") == institution}
        return [tuple(sorted(k)) for k in self.connections if k <= members]


# ================================================================ Neo4j (hosted)

class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        # notifications off: Neo4j otherwise logs a warning for every query that names a property no node has yet
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=240,
                                           notifications_min_severity="OFF")
        self._ready = False

    def _run(self, query: str, **params) -> list[dict]:
        if not self._ready:
            self._setup()
        records, _, _ = self.driver.execute_query(query, params, database_="neo4j")
        return [r.data() for r in records]

    def _setup(self) -> None:
        self._ready = True
        for q in ("CREATE CONSTRAINT person_uid IF NOT EXISTS FOR (p:Person) REQUIRE p.uid IS UNIQUE",
                  "CREATE CONSTRAINT institution_key IF NOT EXISTS FOR (i:Institution) REQUIRE i.key IS UNIQUE",
                  "CREATE CONSTRAINT topic_key IF NOT EXISTS FOR (t:Topic) REQUIRE t.key IS UNIQUE",
                  "CREATE CONSTRAINT method_key IF NOT EXISTS FOR (m:Method) REQUIRE m.key IS UNIQUE"):
            self.driver.execute_query(q, database_="neo4j")

    # ---------------------------------------------------------------- colleges
    def institution(self, key: str) -> dict | None:
        rows = self._run("MATCH (i:Institution {key: $key}) RETURN i {.key, .name, .domains} AS i", key=key)
        return rows[0]["i"] if rows else None

    def upsert_institution(self, name: str, domain: str = "") -> dict:
        rows = self._run(
            "MERGE (i:Institution {key: $key}) ON CREATE SET i.name = $name, i.domains = [] "
            "WITH i SET i.domains = CASE WHEN $domain <> '' AND NOT $domain IN i.domains THEN i.domains + $domain ELSE i.domains END "
            "RETURN i {.key, .name, .domains} AS i", key=key_of(name), name=name.strip(), domain=domain)
        return rows[0]["i"]

    def institution_for_domain(self, domain: str) -> dict | None:
        rows = self._run("MATCH (i:Institution) WHERE any(d IN i.domains WHERE $domain = d OR $domain ENDS WITH '.' + d) "
                         "RETURN i {.key, .name, .domains} AS i LIMIT 1", domain=domain)
        return rows[0]["i"] if rows else None

    def search_institutions(self, q: str, limit: int = 10) -> list[dict]:
        rows = self._run(
            "MATCH (i:Institution) WHERE $q = '' OR i.key CONTAINS $q OR any(d IN i.domains WHERE d CONTAINS $q) "
            "OPTIONAL MATCH (p:Person)-[:MEMBER_OF]->(i) "
            "WITH i, count(p) AS members RETURN i {.key, .name, .domains, members: members} AS i "
            "ORDER BY members DESC, i.name LIMIT $limit", q=key_of(q), limit=limit)
        return [r["i"] for r in rows]

    # ---------------------------------------------------------------- people
    _PERSON = ("p {.uid, .name, .role, .department, .verified, .visibility, .published, .field, .bio, .email_domain, "
               ".updated_at, .joined_at, needs: coalesce(p.needs_json, '[]'), institution: i.key, "
               "topics: [(p)-[w:WORKS_ON]->(t:Topic) | {key: t.key, name: t.name, weight: w.weight}], "
               "methods: [(p)-[u:USES]->(m:Method) | {key: m.key, name: m.name, weight: u.weight}]} AS p")

    @staticmethod
    def _person(row: dict) -> dict:
        p = dict(row["p"])
        p["needs"] = json.loads(p.get("needs") or "[]")
        p["institution"] = p.get("institution") or ""
        for k in ("verified", "published"):
            p[k] = bool(p.get(k))
        return p

    def get_person(self, uid: str) -> dict | None:
        rows = self._run(f"MATCH (p:Person {{uid: $uid}}) OPTIONAL MATCH (p)-[:MEMBER_OF]->(i:Institution) RETURN {self._PERSON}", uid=uid)
        return self._person(rows[0]) if rows else None

    def upsert_person(self, uid: str, **fields) -> dict:
        simple = {k: v for k, v in fields.items() if k in ("name", "role", "department", "verified", "visibility",
                                                             "published", "field", "bio", "email_domain")}
        self._run("MERGE (p:Person {uid: $uid}) ON CREATE SET p.joined_at = $now, p.verified = false, "
                  "p.published = false, p.visibility = 'community' SET p += $simple, p.updated_at = $now",
                  uid=uid, simple=simple, now=now_iso())
        if "needs" in fields:
            self._run("MATCH (p:Person {uid: $uid}) SET p.needs_json = $needs", uid=uid, needs=json.dumps(fields["needs"]))
        if "institution" in fields:
            self._run("MATCH (p:Person {uid: $uid}) OPTIONAL MATCH (p)-[r:MEMBER_OF]->() DELETE r "
                      "WITH p MATCH (i:Institution {key: $key}) MERGE (p)-[:MEMBER_OF]->(i)", uid=uid, key=fields["institution"])
        for field, label, rel in (("topics", "Topic", "WORKS_ON"), ("methods", "Method", "USES")):
            if field in fields:
                items = [{"key": x["key"], "name": x["name"], "weight": float(x.get("weight", 1.0))} for x in fields[field]]
                self._run(f"MATCH (p:Person {{uid: $uid}}) OPTIONAL MATCH (p)-[r:{rel}]->() DELETE r "
                          f"WITH DISTINCT p UNWIND $items AS x MERGE (n:{label} {{key: x.key}}) ON CREATE SET n.name = x.name "
                          f"MERGE (p)-[w:{rel}]->(n) SET w.weight = x.weight", uid=uid, items=items)
        return self.get_person(uid) or {}

    def members(self, institution: str) -> list[dict]:
        rows = self._run(f"MATCH (p:Person)-[:MEMBER_OF]->(i:Institution {{key: $key}}) RETURN {self._PERSON}", key=institution)
        return [self._person(r) for r in rows]

    # ---------------------------------------------------------------- connections
    def request(self, a: str, b: str, message: str) -> None:
        rows = self._run("MATCH (a:Person {uid: $a}), (b:Person {uid: $b}) "
                         "OPTIONAL MATCH (b)-[r:REQUESTED]->(a) RETURN r IS NOT NULL AS mutual, "
                         "EXISTS { (a)-[:CONNECTED]-(b) } AS connected", a=a, b=b)
        if not rows or rows[0]["connected"]:
            return
        if rows[0]["mutual"]:
            self.accept(b, a)
            return
        self._run("MATCH (a:Person {uid: $a}), (b:Person {uid: $b}) MERGE (a)-[r:REQUESTED]->(b) "
                  "SET r.message = $message, r.at = $now", a=a, b=b, message=message[:500], now=now_iso())

    def accept(self, a: str, b: str) -> bool:
        rows = self._run("MATCH (a:Person {uid: $a})-[r:REQUESTED]->(b:Person {uid: $b}) DELETE r "
                         "MERGE (a)-[c:CONNECTED]->(b) ON CREATE SET c.since = $now RETURN count(c) AS n",
                         a=a, b=b, now=now_iso())
        return bool(rows and rows[0]["n"])

    def decline(self, a: str, b: str) -> None:
        self._run("MATCH (:Person {uid: $a})-[r:REQUESTED]->(:Person {uid: $b}) DELETE r", a=a, b=b)

    def disconnect(self, a: str, b: str) -> None:
        self._run("MATCH (x:Person {uid: $a})-[r:CONNECTED|REQUESTED]-(y:Person {uid: $b}) DELETE r", a=a, b=b)

    def network(self, uid: str) -> dict:
        rows = self._run(
            "MATCH (me:Person {uid: $uid}) "
            "RETURN [(me)-[:CONNECTED]-(o) | o.uid] AS connections, "
            "[(o)-[r:REQUESTED]->(me) | {from: o.uid, message: r.message, at: r.at}] AS incoming, "
            "[(me)-[r:REQUESTED]->(o) | {to: o.uid, message: r.message, at: r.at}] AS outgoing, "
            "[(me)-[:COAUTHOR]-(o) | o.uid] AS coauthors", uid=uid)
        if not rows:
            return {"connections": [], "incoming": [], "outgoing": [], "coauthors": []}
        r = rows[0]
        return {"connections": sorted(set(r["connections"])), "incoming": r["incoming"], "outgoing": r["outgoing"],
                "coauthors": sorted(set(r["coauthors"]))}

    def add_coauthors(self, uids: list[str]) -> None:
        self._run("UNWIND $pairs AS pr MATCH (a:Person {uid: pr[0]}), (b:Person {uid: pr[1]}) MERGE (a)-[:COAUTHOR]-(b)",
                  pairs=[[a, b] for i, a in enumerate(uids) for b in uids[i + 1:] if a != b])

    def connections_among(self, institution: str) -> list[tuple[str, str]]:
        rows = self._run("MATCH (a:Person)-[:MEMBER_OF]->(i:Institution {key: $key})<-[:MEMBER_OF]-(b:Person), "
                         "(a)-[:CONNECTED]-(b) WHERE a.uid < b.uid RETURN DISTINCT a.uid AS a, b.uid AS b", key=institution)
        return [(r["a"], r["b"]) for r in rows]


# ================================================================ the one in use

_store = None
_store_lock = threading.Lock()


def store():
    global _store
    with _store_lock:
        if _store is None:
            # the shared graph only behind sign-in: the single-user laptop version keeps its own JSON file
            if settings.neo4j_uri and settings.hosted:
                _store = Neo4jStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
            else:
                _store = MemoryStore(settings.data_dir / "community.json")
        return _store


def use(s) -> None:
    """Swap the store (tests)."""
    global _store
    _store = s
