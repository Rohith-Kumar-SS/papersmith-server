"""Research papers: a college's output (for its admins' dashboard) and a person's own papers (claimed by them).

OpenAlex (free, CC0) covers every college in the directory that has an OpenAlex ID. A college whose admin adds
its own Elsevier key uses Scopus instead. Nobody is imported as a person: authors who haven't signed up are
only names on papers, and people claim their own author profile to show their papers.
"""

from __future__ import annotations

import logging
import time

import httpx

from ..config import settings
from ..models import now_iso
from .store import store

log = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org"
SCOPUS = "https://api.elsevier.com/content/search/scopus"
REFRESH_AFTER = 24 * 3600
MIN_REFRESH_GAP = 600
WORK_FIELDS = "id,doi,display_name,publication_year,cited_by_count,primary_location,primary_topic,topics,authorships"


class SourceError(ValueError):
    pass


def _openalex(path: str, **params) -> dict:
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    if settings.openalex_mailto:
        params["mailto"] = settings.openalex_mailto
    try:
        r = httpx.get(f"{OPENALEX}{path}", params=params, timeout=30,
                      headers={"User-Agent": "PaperSmith research community"})
    except httpx.HTTPError as exc:
        raise SourceError("Couldn't reach OpenAlex right now. Try again later.") from exc
    if r.status_code == 429:
        raise SourceError("OpenAlex's free daily allowance is used up for today. Try again tomorrow.")
    if r.status_code >= 400:
        raise SourceError(f"OpenAlex answered {r.status_code}.")
    return r.json()


def _work(w: dict) -> dict:
    loc = w.get("primary_location") or {}
    venue = ((loc.get("source") or {}).get("display_name")) or ""
    topics = [t["display_name"] for t in (w.get("topics") or [])[:3] if t.get("display_name")]
    if not topics and w.get("primary_topic"):
        topics = [w["primary_topic"]["display_name"]]
    authors = [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])]
    return {"id": (w.get("id") or "").rsplit("/", 1)[-1], "title": w.get("display_name") or "",
            "year": w.get("publication_year") or 0, "venue": venue, "doi": w.get("doi") or "",
            "cited_by": w.get("cited_by_count") or 0, "topics": topics,
            "authors": [a for a in authors if a][:4] + (["…"] if len(authors) > 4 else [])}


# ---------------------------------------------------------------- a college's output

def _openalex_college(openalex_id: str) -> dict:
    inst = f"authorships.institutions.lineage:{openalex_id}"
    this_year = time.gmtime().tm_year
    years = _openalex("/works", filter=f"{inst},from_publication_date:{this_year - 9}-01-01", group_by="publication_year")
    topics = _openalex("/works", filter=f"{inst},from_publication_date:{this_year - 2}-01-01", group_by="primary_topic.id")
    recent = _openalex("/works", filter=inst, sort="publication_date:desc", **{"per-page": 20, "select": WORK_FIELDS})
    cited = _openalex("/works", filter=f"{inst},from_publication_date:{this_year - 3}-01-01", sort="cited_by_count:desc",
                      **{"per-page": 10, "select": WORK_FIELDS})
    by_year = sorted(({"year": int(g["key"]), "count": g["count"]} for g in years.get("group_by", []) if str(g["key"]).isdigit()),
                     key=lambda x: x["year"])
    return {"source": "OpenAlex", "total": sum(y["count"] for y in by_year), "by_year": by_year,
            "topics": [{"name": g["key_display_name"], "count": g["count"]} for g in topics.get("group_by", [])[:15]
                       if g.get("key_display_name") and g["key_display_name"] != "unknown"],
            "recent": [_work(w) for w in recent.get("results", [])],
            "top_cited": [_work(w) for w in cited.get("results", [])]}


def _scopus(params: dict, cfg: dict) -> dict:
    headers = {"X-ELS-APIKey": cfg["api_key"], "Accept": "application/json"}
    if cfg.get("insttoken"):
        headers["X-ELS-Insttoken"] = cfg["insttoken"]
    try:
        r = httpx.get(SCOPUS, params=params, headers=headers, timeout=30)
    except httpx.HTTPError as exc:
        raise SourceError("Couldn't reach Scopus right now.") from exc
    if r.status_code in (401, 403):
        raise SourceError("Scopus refused the key. Check the API key and institution token, and that your college "
                          "subscribes to Scopus.")
    if r.status_code == 429:
        raise SourceError("Scopus's weekly allowance for this key is used up.")
    if r.status_code >= 400:
        raise SourceError(f"Scopus answered {r.status_code}.")
    return r.json().get("search-results", {})


def _scopus_college(cfg: dict) -> dict:
    afid = cfg["afid"]
    this_year = time.gmtime().tm_year
    fields = "dc:title,prism:coverDate,prism:publicationName,prism:doi,citedby-count,dc:creator,eid"
    recent = _scopus({"query": f"AF-ID({afid})", "sort": "-coverDate", "count": 20, "field": fields}, cfg)
    cited = _scopus({"query": f"AF-ID({afid}) AND PUBYEAR > {this_year - 4}", "sort": "-citedby-count", "count": 10,
                     "field": fields}, cfg)
    by_year = []
    for y in range(this_year - 9, this_year + 1):
        res = _scopus({"query": f"AF-ID({afid}) AND PUBYEAR = {y}", "count": 1, "field": "eid"}, cfg)
        by_year.append({"year": y, "count": int(res.get("opensearch:totalResults", 0) or 0)})

    def work(e: dict) -> dict:
        return {"id": e.get("eid", ""), "title": e.get("dc:title", ""), "year": int((e.get("prism:coverDate") or "0")[:4] or 0),
                "venue": e.get("prism:publicationName", ""), "doi": f"https://doi.org/{e['prism:doi']}" if e.get("prism:doi") else "",
                "cited_by": int(e.get("citedby-count", 0) or 0), "topics": [], "authors": [e.get("dc:creator", "")] if e.get("dc:creator") else []}

    return {"source": "Scopus", "total": int(recent.get("opensearch:totalResults", 0) or 0), "by_year": by_year, "topics": [],
            "recent": [work(e) for e in recent.get("entry", []) if "error" not in e],
            "top_cited": [work(e) for e in cited.get("entry", []) if "error" not in e]}


def college_output(inst: dict, refresh: bool = False) -> dict:
    """The college's papers, cached for a day: {source, total, by_year, topics, recent, top_cited, fetched_at}."""
    s = store()
    cached = s.get_doc("pubcache", inst["key"])
    age = time.time() - (cached or {}).get("fetched_ts", 0)
    if cached and (age < REFRESH_AFTER and not refresh or refresh and age < MIN_REFRESH_GAP):
        return cached
    scopus = inst.get("scopus") or {}
    if scopus.get("api_key") and scopus.get("afid"):
        data = _scopus_college(scopus)
    elif inst.get("openalex"):
        data = _openalex_college(inst["openalex"])
    else:
        return {"source": "", "total": 0, "by_year": [], "topics": [], "recent": [], "top_cited": [],
                "missing": "This college has no OpenAlex ID yet. Its admin can add one (or a Scopus key) under Data sources."}
    doc = {"id": inst["key"], "institution": inst["key"], **data, "fetched_at": now_iso(), "fetched_ts": time.time()}
    s.put_doc("pubcache", doc)
    return doc


# ---------------------------------------------------------------- a person's own papers

def find_authors(name: str, openalex_institution: str = "", orcid: str = "") -> list[dict]:
    """Author profiles that could be this person: by ORCID, or by name (at their college first)."""
    if orcid:
        orcid = orcid.strip().rsplit("/", 1)[-1]
        data = _openalex("/authors", filter=f"orcid:{orcid}")
    else:
        flt = f"affiliations.institution.id:{openalex_institution}" if openalex_institution else None
        params = {"search": name, "per-page": 8}
        if flt:
            params["filter"] = flt
        data = _openalex("/authors", **params)
        if not data.get("results") and flt:
            data = _openalex("/authors", search=name, **{"per-page": 8})
    out = []
    for a in data.get("results", []):
        inst = (a.get("last_known_institutions") or [{}])[0] if a.get("last_known_institutions") else {}
        out.append({"id": a["id"].rsplit("/", 1)[-1], "name": a.get("display_name", ""), "orcid": a.get("orcid") or "",
                    "works_count": a.get("works_count", 0), "cited_by": a.get("cited_by_count", 0),
                    "institution": inst.get("display_name", ""),
                    "topics": [t["display_name"] for t in (a.get("topics") or [])[:4]]})
    return out


def author_works(author_id: str, limit: int = 50) -> tuple[dict, list[dict]]:
    author = _openalex(f"/authors/{author_id}")
    works = _openalex("/works", filter=f"author.id:{author_id}", sort="publication_date:desc",
                      **{"per-page": limit, "select": WORK_FIELDS})
    return author, [_work(w) for w in works.get("results", [])]
