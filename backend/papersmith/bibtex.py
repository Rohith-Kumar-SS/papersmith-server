"""Minimal BibTeX reader/writer (no external dependency)."""

from __future__ import annotations

import re

from .models import Reference

_ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.S)


def _balanced(text: str, start: int) -> int:
    """Index just past the brace that closes the one opened before `start`."""
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def _fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    field_re = re.compile(r"\s*(\w[\w-]*)\s*=\s*", re.S)
    while i < len(body):
        m = field_re.match(body, i)
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        j = m.end()
        if j < len(body) and body[j] == "{":
            end = _balanced(body, j + 1)
            value = body[j + 1 : end - 1]
            i = end
        elif j < len(body) and body[j] == '"':
            end = body.find('"', j + 1)
            end = len(body) if end == -1 else end
            value = body[j + 1 : end]
            i = end + 1
        else:
            end = body.find(",", j)
            end = len(body) if end == -1 else end
            value = body[j:end]
            i = end
        fields[name] = re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()
        comma = body.find(",", i)
        i = len(body) if comma == -1 else comma + 1
    return fields


def parse(text: str) -> list[Reference]:
    refs: list[Reference] = []
    for m in _ENTRY_START.finditer(text):
        kind = m.group(1).lower()
        if kind in ("comment", "string", "preamble"):
            continue
        end = _balanced(text, m.end())
        raw = text[m.start() : end]
        f = _fields(text[m.end() : end - 1])
        refs.append(Reference(
            key=m.group(2),
            title=f.get("title", ""),
            authors=f.get("author", ""),
            year=f.get("year", ""),
            venue=f.get("journal") or f.get("booktitle") or f.get("publisher", ""),
            raw_bibtex=raw.strip(),
        ))
    return refs


def to_bibtex(ref: Reference) -> str:
    if ref.raw_bibtex:
        return ref.raw_bibtex
    fields = {"title": ref.title, "author": ref.authors, "year": ref.year, "howpublished": ref.venue}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@misc{{{ref.key},\n{body}\n}}"
