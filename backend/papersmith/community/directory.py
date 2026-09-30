"""The directory of Indian universities, colleges and research institutions, for the sign-up dropdown.

Built by deploy/build_directory.py from AISHE (Ministry of Education) and OpenAlex. Everyone picks their
college from this list, so one college is one node in the graph whatever people call it ("NIT Trichy",
"National Institute of Technology, Tiruchirappalli"). A college that isn't listed can still be entered; it is
marked unlisted until the owner confirms it.
"""

from __future__ import annotations

import gzip
import json
import re
import threading
from pathlib import Path

DATA = Path(__file__).with_name("data") / "india.json.gz"
KINDS = {"U": "University", "C": "College", "S": "Institute", "R": "Research institute", "X": "Unlisted"}

# what people type -> what the directory says
CITY_ALIASES = {
    "trichy": "tiruchirappalli", "tiruchi": "tiruchirappalli", "bangalore": "bengaluru", "bombay": "mumbai",
    "madras": "chennai", "calcutta": "kolkata", "pondicherry": "puducherry", "pondy": "puducherry",
    "vizag": "visakhapatnam", "baroda": "vadodara", "poona": "pune", "mysore": "mysuru", "cochin": "kochi",
    "calicut": "kozhikode", "trivandrum": "thiruvananthapuram", "mangalore": "mangaluru", "gurgaon": "gurugram",
    "benares": "varanasi", "banaras": "varanasi", "allahabad": "prayagraj", "belgaum": "belagavi",
    "hubli": "hubballi", "gulbarga": "kalaburagi", "tanjore": "thanjavur", "kovai": "coimbatore",
}
WORD_ALIASES = {"govt": "government", "engg": "engineering", "engg.": "engineering", "inst": "institute",
                "univ": "university", "coll": "college", "sci": "science", "mgmt": "management", "tech": "technology",
                "&": "and", "st": "saint"}
_STOP = {"the", "of", "and", "at", "in", "for", "a", "an"}

DEPARTMENTS = [
    # engineering and technology
    "Computer Science and Engineering", "Information Technology", "Artificial Intelligence and Data Science",
    "Artificial Intelligence and Machine Learning", "Cyber Security", "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering", "Electrical Engineering", "Electronics and Instrumentation Engineering",
    "Instrumentation and Control Engineering", "Mechanical Engineering", "Mechatronics", "Automobile Engineering",
    "Aeronautical Engineering", "Aerospace Engineering", "Civil Engineering", "Chemical Engineering",
    "Biotechnology", "Biomedical Engineering", "Metallurgical and Materials Engineering", "Production Engineering",
    "Industrial Engineering", "Agricultural Engineering", "Food Technology", "Textile Technology",
    "Petroleum Engineering", "Mining Engineering", "Marine Engineering", "Environmental Engineering",
    "Architecture", "Planning",
    # sciences
    "Physics", "Chemistry", "Mathematics", "Statistics", "Computer Applications", "Computer Science",
    "Data Science", "Biology", "Botany", "Zoology", "Microbiology", "Biochemistry", "Genetics",
    "Environmental Science", "Geology", "Earth Sciences", "Geography", "Electronics", "Nanotechnology",
    # health
    "Medicine", "Surgery", "Community Medicine", "Nursing", "Pharmacy", "Pharmaceutical Sciences", "Dentistry",
    "Physiotherapy", "Public Health", "Ayurveda", "Homoeopathy", "Allied Health Sciences",
    # agriculture
    "Agriculture", "Horticulture", "Veterinary Science", "Fisheries Science", "Forestry",
    # business, law, humanities
    "Management Studies", "Commerce", "Economics", "Finance", "Law", "English", "Hindi", "Tamil", "Languages",
    "History", "Political Science", "Public Administration", "Sociology", "Psychology", "Philosophy",
    "Social Work", "Education", "Journalism and Mass Communication", "Library and Information Science",
    "Fine Arts", "Design", "Music", "Physical Education", "Hotel Management",
    # other
    "Research Office", "Administration", "Other",
]

_lock = threading.Lock()
_rows: list[list] | None = None
_states: list[str] = []
_hay: list[str] = []
_acr: list[str] = []
_by_id: dict[str, int] = {}
_by_domain: dict[str, int] = {}


def norm(text: str) -> str:
    """'Govt. Engg. College, Trichy' -> 'government engineering college tiruchirappalli'."""
    words = re.findall(r"[a-z0-9&]+", text.lower().replace("&", " & "))
    out = []
    for w in words:
        w = WORD_ALIASES.get(w, w)
        w = CITY_ALIASES.get(w, w)
        if w not in _STOP:
            out.append(w)
    return " ".join(out)


def acronym(name: str) -> str:
    """'National Institute of Technology Tiruchirappalli' -> 'nitt' (for people who type 'NIT Trichy', 'NITT')."""
    core = re.sub(r"\s*[,(].*$", "", name)
    words = [w for w in re.findall(r"[A-Za-z]+", core) if w.lower() not in _STOP]
    return "".join(w[0] for w in words).lower() if len(words) > 1 else ""


def _load() -> None:
    global _rows, _states
    with _lock:
        if _rows is not None:
            return
        if not DATA.exists():
            _rows, _states = [], []
            return
        with gzip.open(DATA, "rt", encoding="utf-8") as f:
            data = json.load(f)
        _states = data["states"]
        rows = data["rows"]
        for i, r in enumerate(rows):
            _id, name, _kind, state, district, _oa, _ror, domain, _works, aka = r
            # a leading space so a typed word matches the start of a word: " nit" is not in " community"
            _hay.append(" " + " ".join([norm(name), norm(district), norm(_states[state]), *(norm(a) for a in aka)]))
            short = {a.replace(".", "").lower() for a in aka
                     if " " not in a and len(a) <= 10 and sum(ch.isupper() for ch in a) >= 2}      # 'IISc', 'NITT'
            _acr.append(" " + " ".join(sorted({acronym(name), *short} - {""})) + " ")
            _by_id[_id] = i
            if domain:
                _by_domain.setdefault(domain, i)
        _rows = rows


def _view(i: int) -> dict:
    _id, name, kind, state, district, oa, ror, domain, works, aka = _rows[i]  # type: ignore[index]
    return {"id": _id, "name": name, "kind": kind, "kind_label": KINDS.get(kind, ""), "state": _states[state],
            "district": district, "openalex": oa, "ror": ror, "domain": domain, "works": works,
            "place": ", ".join(x for x in (district, _states[state]) if x)}


def size() -> int:
    _load()
    return len(_rows or [])


def get(inst_id: str) -> dict | None:
    _load()
    i = _by_id.get(inst_id)
    return _view(i) if i is not None else None


def for_domain(domain: str) -> dict | None:
    """The listed institution whose website domain an email address belongs to (cs.nitt.edu -> nitt.edu)."""
    _load()
    parts = domain.lower().split(".")
    for k in range(len(parts) - 1):
        i = _by_domain.get(".".join(parts[k:]))
        if i is not None:
            return _view(i)
    return None


def search(q: str, limit: int = 12, boost: dict[str, int] | None = None) -> list[dict]:
    """Institutions matching every word typed (in the name, its acronym, district or state), best first:
    colleges that already have members, then research-active ones, then shorter names."""
    _load()
    words = norm(q).split()
    if not words or not _rows:
        return []
    boost = boost or {}
    whole = norm(q)
    found = []
    for i, hay in enumerate(_hay):
        acr = _acr[i]
        if all(" " + w in hay or (len(w) >= 2 and " " + w in acr) for w in words):
            r = _rows[i]
            name = norm(r[1])
            score = boost.get(r[0], 0) * 1000 + (500 if name == whole else 0) + (200 if name.startswith(whole) else 0)
            score += min(300, (r[8] or 0) ** 0.5) + (40 if r[2] == "U" else 0) - len(name) * 0.2
            if len(words) == 1 and f" {words[0]} " in acr:
                score += 600                # typed the exact acronym: 'nitt', 'iisc'
            elif any(f" {w} " in acr for w in words):
                score += 150                # 'nit trichy'

            found.append((score, i))
    found.sort(key=lambda x: -x[0])
    return [_view(i) for _, i in found[:limit]]
