"""Read uploaded research material into labelled passages, tables and figures.

Every passage keeps a human-readable location ("slide 4", "p. 3", "§ Methods ¶ 2") so a claim
extracted from it can show the author exactly where it came from.

Supported: PDF (text layer), PPTX (slides, notes, tables, charts, pictures), DOCX (paragraphs,
tables), TXT/MD, CSV/XLSX (tables), BibTeX (references), PNG/JPG (figures).
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

MAX_PASSAGE_CHARS = 1400      # longer blocks are split at sentence boundaries
MAX_TABLE_ROWS = 40
MAX_TABLE_COLS = 12

KIND_BY_EXT = {
    ".pdf": "pdf", ".pptx": "pptx", ".docx": "docx", ".txt": "txt", ".md": "md", ".markdown": "md",
    ".csv": "csv", ".tsv": "csv", ".xlsx": "xlsx", ".bib": "bib",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
}
SUPPORTED = ", ".join(sorted({e for e in KIND_BY_EXT}))


class IngestError(Exception):
    """A file that cannot be read, with a message meant for the author."""


@dataclass
class ParsedTable:
    caption: str
    columns: list[str]
    rows: list[list[str]]
    location: str


@dataclass
class ParsedImage:
    data: bytes
    ext: str
    caption: str
    location: str


@dataclass
class ParsedDoc:
    kind: str
    unit_label: str                 # page | slide | paragraph | sheet | file
    units: int = 0
    passages: list[tuple[str, str]] = field(default_factory=list)   # (location, text)
    tables: list[ParsedTable] = field(default_factory=list)
    images: list[ParsedImage] = field(default_factory=list)
    title: str = ""
    meta: dict = field(default_factory=dict)                         # authors, year, doi (reference papers)
    bibtex: str = ""


def kind_of(filename: str) -> str:
    return KIND_BY_EXT.get(Path(filename).suffix.lower(), "other")


# ---------------------------------------------------------------- text helpers

def _clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_long(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sent) + 1 > limit:
            parts.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent
    if current.strip():
        parts.append(current.strip())
    # a single sentence longer than the limit is cut hard
    out = []
    for p in parts:
        out += [p[i : i + limit] for i in range(0, len(p), limit)] if len(p) > limit else [p]
    return out


def _add(doc: ParsedDoc, location: str, text: str) -> None:
    text = _clean(text)
    if len(re.sub(r"\W", "", text)) < 12:      # page numbers, stray symbols
        return
    pieces = _split_long(text)
    for i, piece in enumerate(pieces, start=1):
        loc = location if len(pieces) == 1 else f"{location} ({i}/{len(pieces)})"
        doc.passages.append((loc, piece))


def _table_from_rows(rows: list[list], caption: str, location: str) -> ParsedTable | None:
    rows = [[("" if v is None else str(v)).strip() for v in r][:MAX_TABLE_COLS] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return ParsedTable(caption=caption, columns=rows[0], rows=rows[1 : MAX_TABLE_ROWS + 1], location=location)


def table_as_text(t: ParsedTable) -> str:
    """Linear form of a table, so its numbers can support claims."""
    lines = [f"{t.caption}."] if t.caption else []
    for r in t.rows:
        lines.append("; ".join(f"{c} = {v}" for c, v in zip(t.columns, r) if v))
    return " ".join(lines)


# ---------------------------------------------------------------- PDF

_REF_HEADING = re.compile(r"^\s*(references|bibliography|literature cited|works cited)\s*$", re.I | re.M)


def _pdf_meta(first_page: str, reader) -> dict:
    meta: dict = {}
    info = getattr(reader, "metadata", None) or {}
    title = (info.get("/Title") or "").strip() if hasattr(info, "get") else ""
    author = (info.get("/Author") or "").strip() if hasattr(info, "get") else ""
    lines = [l.strip() for l in first_page.splitlines() if l.strip()]
    if not title or len(title) < 8 or title.lower().endswith((".doc", ".docx", ".pdf", ".tex")):
        # first reasonably long line near the top is almost always the title
        title = next((l for l in lines[:8] if 15 <= len(l) <= 220 and not re.match(r"^(arxiv|doi|http)", l, re.I)), "")
    meta["title"] = title
    if not author and title in lines:
        # the author line usually follows the title: names separated by commas / "and", no digits
        for cand in lines[lines.index(title) + 1 : lines.index(title) + 4]:
            if (5 <= len(cand) <= 200 and not re.search(r"\d|@|journal|university|institute|abstract|doi|http", cand, re.I)
                    and re.search(r"[A-Z][a-z]+", cand) and (", " in cand or " and " in cand or len(cand.split()) <= 4)):
                author = cand
                break
    meta["authors"] = author
    doi = re.search(r"\b10\.\d{4,9}/[^\s\"<>]+", first_page)
    if doi:
        meta["doi"] = doi.group(0).rstrip(".,;)")
    years = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", first_page[:3000])
    if years:
        meta["year"] = Counter(years).most_common(1)[0][0]
    return meta


def parse_pdf(data: bytes) -> ParsedDoc:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                raise IngestError("This PDF is password-protected. Remove the password and upload it again.") from None
        pages = [(p.extract_text() or "") for p in reader.pages]
    except IngestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"This PDF could not be read ({type(exc).__name__}).") from exc

    doc = ParsedDoc(kind="pdf", unit_label="page", units=len(pages))
    if sum(len(p.strip()) for p in pages) < max(40, 20 * len(pages)):   # a scan has (almost) no text layer
        raise IngestError("This PDF has no selectable text; it is probably a scan. Upload a text-based PDF, "
                          "or paste the text into the chat.")

    # lines repeated on many pages are running headers / footers
    line_counts: Counter = Counter()
    for p in pages:
        line_counts.update({l.strip() for l in p.splitlines() if l.strip()})
    repeated = {l for l, n in line_counts.items() if len(pages) >= 4 and n >= 0.5 * len(pages)}

    doc.meta = _pdf_meta(pages[0] if pages else "", reader)
    doc.title = doc.meta.get("title", "")
    for no, page in enumerate(pages, start=1):
        text = "\n".join(l for l in page.splitlines() if l.strip() not in repeated)
        m = _REF_HEADING.search(text)
        stop = m is not None and no > 1
        if m:
            text = text[: m.start()]
        # blocks: blank lines, or a line ending a sentence followed by a capitalised line
        blocks = re.split(r"\n\s*\n|(?<=[.!?:])\n(?=[A-Z0-9•\-])", text)
        for block in blocks:
            block = re.sub(r"\s*\n\s*", " ", block)
            _add(doc, f"p. {no}", block)
        if stop:
            break                                 # never mine the reference list for claims
    return doc


# ---------------------------------------------------------------- PPTX

def parse_pptx(data: bytes) -> ParsedDoc:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    try:
        prs = Presentation(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"This PowerPoint file could not be read ({type(exc).__name__}). Save it as .pptx and try again.") from exc

    doc = ParsedDoc(kind="pptx", unit_label="slide", units=len(prs.slides))

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from walk(sh.shapes)
            else:
                yield sh

    for no, slide in enumerate(prs.slides, start=1):
        loc = f"slide {no}"
        title = ""
        try:
            if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
                title = slide.shapes.title.text_frame.text.strip()
        except Exception:  # noqa: BLE001
            title = ""
        if no == 1 and title:
            doc.title = title
        lines: list[str] = []
        for sh in walk(slide.shapes):
            try:
                if sh.has_text_frame and sh.text_frame.text.strip() and sh.text_frame.text.strip() != title:
                    for para in sh.text_frame.paragraphs:
                        t = "".join(r.text for r in para.runs).strip()
                        if t:
                            lines.append(("  " * para.level) + "- " + t)
                if getattr(sh, "has_table", False) and sh.has_table:
                    rows = [[c.text for c in r.cells] for r in sh.table.rows]
                    t = _table_from_rows(rows, title or f"Table on slide {no}", loc)
                    if t:
                        doc.tables.append(t)
                if getattr(sh, "has_chart", False) and sh.has_chart:
                    chart = sh.chart
                    plot = chart.plots[0]
                    cats = [str(c) for c in plot.categories]
                    rows = [["Series"] + cats] + [[s.name] + [f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in s.values]
                                                  for s in plot.series]
                    cap = (chart.chart_title.text_frame.text if chart.has_title else "") or title or f"Chart on slide {no}"
                    t = _table_from_rows(rows, cap, loc)
                    if t:
                        doc.tables.append(t)
                if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    img = sh.image
                    if len(img.blob) > 8_000:                       # skip logos and icons
                        doc.images.append(ParsedImage(img.blob, "." + img.ext.lower(), title or f"Figure from slide {no}", loc))
            except Exception:  # noqa: BLE001 - one odd shape must not sink the deck
                continue
        notes = ""
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:  # noqa: BLE001
            notes = ""
        body = (f"{title}\n" if title else "") + "\n".join(lines)
        if notes:
            body += f"\nSpeaker notes: {notes}"
        _add(doc, loc, body.replace("\n", " \n "))
    return doc


# ---------------------------------------------------------------- DOCX

def parse_docx(data: bytes) -> ParsedDoc:
    import docx

    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"This Word file could not be read ({type(exc).__name__}). Save it as .docx and try again.") from exc

    doc = ParsedDoc(kind="docx", unit_label="paragraph")
    heading = ""
    count = 0
    buffer: list[str] = []
    buffer_start = 0

    def flush():
        nonlocal buffer
        if buffer:
            where = f"§ {heading} " if heading else ""
            _add(doc, f"{where}¶ {buffer_start}".strip(), " ".join(buffer))
            buffer = []

    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading") or style == "title":
            flush()
            if style == "title" and not doc.title:
                doc.title = text
            heading = text[:60]
            continue
        count += 1
        if not buffer:
            buffer_start = count
        buffer.append(text)
        if sum(len(b) for b in buffer) > 700:
            flush()
    flush()
    doc.units = count
    for i, table in enumerate(d.tables, start=1):
        rows = [[c.text for c in r.cells] for r in table.rows]
        t = _table_from_rows(rows, f"Table {i}", f"table {i}")
        if t:
            doc.tables.append(t)
    if not doc.title and doc.passages:
        doc.title = doc.passages[0][1][:120]
    return doc


# ---------------------------------------------------------------- plain text / markdown

def parse_text(data: bytes, kind: str) -> ParsedDoc:
    text = data.decode("utf-8", errors="replace")
    doc = ParsedDoc(kind=kind, unit_label="paragraph")
    heading = ""
    for no, block in enumerate(re.split(r"\n\s*\n", text), start=1):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", block.splitlines()[0])
        if m:
            heading = m.group(2).strip()[:60]
            if not doc.title and len(m.group(1)) == 1:
                doc.title = heading
            block = "\n".join(block.splitlines()[1:]).strip()
            if not block:
                continue
        doc.units += 1
        where = f"§ {heading} " if heading else ""
        _add(doc, f"{where}¶ {doc.units}".strip(), re.sub(r"\s*\n\s*", " ", block))
    return doc


# ---------------------------------------------------------------- tables

def parse_csv(data: bytes, name: str) -> ParsedDoc:
    text = data.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text[:4000], delimiters=",;\t") if text.strip() else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    doc = ParsedDoc(kind="csv", unit_label="table", units=1)
    t = _table_from_rows(rows, Path(name).stem.replace("_", " "), "table")
    if not t:
        raise IngestError("This file has no table rows to read.")
    doc.tables.append(t)
    doc.passages.append(("table", table_as_text(t)))
    return doc


def parse_xlsx(data: bytes, name: str) -> ParsedDoc:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"This Excel file could not be read ({type(exc).__name__}).") from exc
    doc = ParsedDoc(kind="xlsx", unit_label="sheet")
    for ws in wb.worksheets:
        rows = [list(r) for _, r in zip(range(MAX_TABLE_ROWS + 1), ws.iter_rows(values_only=True))]
        rows = [[f"{v:g}" if isinstance(v, float) else v for v in r] for r in rows]
        t = _table_from_rows(rows, f"{Path(name).stem} – {ws.title}", f"sheet {ws.title}")
        if t:
            doc.units += 1
            doc.tables.append(t)
            doc.passages.append((f"sheet {ws.title}", table_as_text(t)))
    if not doc.tables:
        raise IngestError("This workbook has no sheets with a header row and data.")
    return doc


# ---------------------------------------------------------------- entry point

def parse(filename: str, data: bytes) -> ParsedDoc:
    kind = kind_of(filename)
    if kind == "pdf":
        return parse_pdf(data)
    if kind == "pptx":
        return parse_pptx(data)
    if kind == "docx":
        return parse_docx(data)
    if kind in ("txt", "md"):
        return parse_text(data, kind)
    if kind == "csv":
        return parse_csv(data, filename)
    if kind == "xlsx":
        return parse_xlsx(data, filename)
    if kind == "bib":
        doc = ParsedDoc(kind="bib", unit_label="entry")
        doc.bibtex = data.decode("utf-8", errors="replace")
        return doc
    if kind == "image":
        doc = ParsedDoc(kind="image", unit_label="image", units=1)
        doc.images.append(ParsedImage(data, Path(filename).suffix.lower(), Path(filename).stem.replace("_", " "), "image"))
        return doc
    if filename.lower().endswith((".ppt", ".doc", ".xls")):
        raise IngestError("Old Office formats (.ppt, .doc, .xls) are not supported. Save the file as .pptx, .docx or .xlsx.")
    raise IngestError(f"PaperSmith cannot read this file type. Supported: {SUPPORTED}.")
