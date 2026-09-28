"""Core data model.

The Idea Ledger is the only source of scientific content. Everything the
writer produces must trace back to ledger items by ID, and every ledger item
traces back to the author's own material: a passage of an uploaded file, or a
chat message the author wrote.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClaimType(str, Enum):
    background = "background"        # prior knowledge the author wants stated (usually cited)
    gap = "gap"                      # what is missing in prior work
    objective = "objective"          # what this work sets out to do
    contribution = "contribution"    # what this work contributes
    method = "method"                # a method / procedure step
    dataset = "dataset"              # data used
    result = "result"                # an observed result
    comparison = "comparison"        # result relative to a baseline
    interpretation = "interpretation"  # the author's own reading of a result
    limitation = "limitation"
    future_work = "future_work"


class Claim(BaseModel):
    id: str                                   # "C1", "C2", ...
    type: ClaimType
    text: str
    refs: list[str] = Field(default_factory=list)     # citation keys that support this claim
    tables: list[str] = Field(default_factory=list)   # table IDs this claim draws on
    figures: list[str] = Field(default_factory=list)  # figure IDs this claim draws on
    # where the claim came from: passage IDs of uploaded files ("F2.p7") or chat messages ("chat:M12")
    sources: list[str] = Field(default_factory=list)
    # "suggested": proposed by the mentor and accepted by the author in chat (the receipt shows both messages)
    origin: Literal["file", "chat", "manual", "suggested"] = "manual"


class Reference(BaseModel):
    key: str
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    raw_bibtex: str = ""


class DataTable(BaseModel):
    id: str                                   # "T1"
    caption: str
    columns: list[str]
    rows: list[list[str]]
    source: str = ""                          # uploaded file ID it came from
    location: str = ""                        # "slide 5", "sheet Results"


class Figure(BaseModel):
    id: str                                   # "Fig1"
    caption: str
    filename: str = ""
    source: str = ""
    location: str = ""


class Ledger(BaseModel):
    title: str = "Untitled paper"
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    template: Literal["ieee", "article", "acm"] = "ieee"
    claims: list[Claim] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    tables: list[DataTable] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)

    def claim_map(self) -> dict[str, Claim]:
        return {c.id: c for c in self.claims}

    def ref_map(self) -> dict[str, Reference]:
        return {r.key: r for r in self.references}

    def table_map(self) -> dict[str, DataTable]:
        return {t.id: t for t in self.tables}


# ---------------------------------------------------------------- outline

class OutlineParagraph(BaseModel):
    id: str                                   # "P1"
    claim_ids: list[str]
    stale: bool = False                       # claims changed since it was written
    style: str = ""                           # the author's wording instruction ("more formal", "shorter")


class OutlineSection(BaseModel):
    name: str                                 # "Introduction"
    paragraphs: list[OutlineParagraph] = Field(default_factory=list)
    target_words: int = 0


class Outline(BaseModel):
    sections: list[OutlineSection] = Field(default_factory=list)
    approved: bool = False


# ---------------------------------------------------------------- draft

Severity = Literal["error", "warning", "info"]


class Flag(BaseModel):
    kind: str          # e.g. "number_not_in_source", "invented_citation", "new_entity"
    severity: Severity
    detail: str


class Sentence(BaseModel):
    id: str                                   # "S12"
    text: str
    claim_ids: list[str] = Field(default_factory=list)
    ref_keys: list[str] = Field(default_factory=list)
    signpost: bool = False                    # purely structural sentence, no claims
    flags: list[Flag] = Field(default_factory=list)
    entailment: float | None = None           # NLI P(entailment) against its sources
    contradiction: float | None = None
    status: Literal["verified", "flagged", "user_edited", "accepted"] = "verified"
    origin: Literal["ai", "human"] = "ai"


class DraftParagraph(BaseModel):
    id: str                                   # matches OutlineParagraph.id
    section: str
    claim_ids: list[str]
    sentences: list[Sentence] = Field(default_factory=list)
    issues: list[Flag] = Field(default_factory=list)       # paragraph-level (coverage, dropped citations)
    attempts: int = 0


class Draft(BaseModel):
    paragraphs: list[DraftParagraph] = Field(default_factory=list)
    backend: str = ""
    model: str = ""
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class ConsistencyIssue(BaseModel):
    kind: str
    severity: Severity
    detail: str
    locations: list[str] = Field(default_factory=list)   # sentence / claim / table IDs


class SourceFile(BaseModel):
    """A file the author uploaded. Its text lives in passages (stored separately, see sources.py)."""
    id: str                                   # "D1"
    filename: str
    kind: str                                 # pdf | pptx | docx | txt | md | csv | xlsx | bib | image | other
    role: Literal["own", "reference", "data"] = "own"
    role_set: bool = False                    # the author chose the role (otherwise it was guessed)
    status: Literal["queued", "reading", "extracting", "ready", "error"] = "queued"
    detail: str = ""                          # progress or error message
    units: int = 0                            # pages, slides or paragraphs read
    unit_label: str = "page"
    passages: int = 0
    claims: int = 0
    tables: int = 0
    figures: int = 0
    reference_key: str = ""                   # set when role == "reference"
    stored_as: str = ""
    size: int = 0
    uploaded_at: str = Field(default_factory=now_iso)


class Passage(BaseModel):
    id: str                                   # "D1.p3", "D1.t1" (a table) or "chat:M7"
    source_id: str                            # "D1" or "chat"
    location: str                             # "slide 4", "p. 3", "¶ 12", "message 7"
    text: str


class ChatMessage(BaseModel):
    id: str                                   # "M1"
    role: Literal["user", "assistant"]
    text: str
    kind: Literal["text", "question", "summary", "plan", "progress", "done", "error"] = "text"
    created_at: str = Field(default_factory=now_iso)
    attachments: list[str] = Field(default_factory=list)   # source file IDs
    options: list[str] = Field(default_factory=list)       # quick replies
    question_id: str | None = None
    data: dict = Field(default_factory=dict)               # structured payload for cards


class OpenQuestion(BaseModel):
    id: str                                   # "Q1"
    need: str                                 # "gap", "dataset", "title", ...
    section: str
    text: str
    status: Literal["open", "answered", "skipped"] = "open"
    asked_in: str = ""                        # message ID


class PaperSettings(BaseModel):
    target_pages: int = 6
    venue: str = ""


class Project(BaseModel):
    id: str
    name: str
    owner: str = ""                           # hosted version: the signed-in person the paper belongs to
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    ledger: Ledger = Field(default_factory=Ledger)
    outline: Outline | None = None
    draft: Draft | None = None
    consistency: list[ConsistencyIssue] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)    # audit trail of edits / generations
    # conversational workspace
    settings: PaperSettings = Field(default_factory=PaperSettings)
    sources: list[SourceFile] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    questions: list[OpenQuestion] = Field(default_factory=list)
    stage: Literal["welcome", "collecting", "questions", "ready", "writing", "revising"] = "welcome"
