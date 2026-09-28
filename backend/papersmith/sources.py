"""Passage store: the text of everything the author gave us, with locations, plus lexical search.

Passages live in data/projects/<id>.sources.json (kept apart from the project file, which stays small).
Chat messages the author writes are stored as passages too ("chat:M7"), so a claim taken from a chat
answer has a receipt like any other.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from . import cloud
from . import textutils as tu
from .config import settings
from .models import Passage
from .storage import _lock


def _path(project_id: str) -> Path:
    if not project_id.isalnum():
        raise ValueError("invalid project id")
    return settings.projects_dir / f"{project_id}.sources.json"


def load(project_id: str) -> list[Passage]:
    path = _path(project_id)
    if not path.exists() and cloud.enabled():
        with _lock:
            if not path.exists():
                data = cloud.store().get_passages(project_id)
                if data is not None:
                    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if not path.exists():
        return []
    try:
        return [Passage.model_validate(p) for p in json.loads(path.read_text(encoding="utf-8"))]
    except (OSError, ValueError):
        return []


def save(project_id: str, passages: list[Passage]) -> None:
    with _lock:
        path = _path(project_id)
        rows = [p.model_dump() for p in passages]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        if cloud.enabled():
            cloud.store().put_passages(project_id, rows)


def add(project_id: str, new: list[Passage]) -> None:
    with _lock:
        existing = {p.id: p for p in load(project_id)}
        for p in new:
            existing[p.id] = p
        save(project_id, list(existing.values()))


def remove_source(project_id: str, source_id: str) -> None:
    with _lock:
        save(project_id, [p for p in load(project_id) if p.source_id != source_id])


def delete_all(project_id: str) -> None:
    with _lock:
        _path(project_id).unlink(missing_ok=True)       # the cloud copy goes with the project (storage.delete)


def by_id(project_id: str) -> dict[str, Passage]:
    return {p.id: p for p in load(project_id)}


# ---------------------------------------------------------------- search (BM25)

def _terms(text: str) -> list[str]:
    return [tu.stem(w) for w in tu.words(text) if w.lower() not in tu.STOPWORDS and len(w) > 2] + \
           [n for n in re.findall(r"\d+(?:\.\d+)?", text)]


def search(passages: list[Passage], query: str, k: int = 6, k1: float = 1.4, b: float = 0.75) -> list[tuple[Passage, float]]:
    q = set(_terms(query))
    if not q or not passages:
        return []
    docs = [_terms(p.text + " " + p.location) for p in passages]
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scored = []
    for p, d in zip(passages, docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        if s > 0:
            scored.append((p, s))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
