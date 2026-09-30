"""HTTP API. Run with:  uvicorn papersmith.api:app --port 8765 --reload"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, auth, bibtex, consistency, export, jobs, planner, provenance, sources, storage, verifier, workspace, writer
from . import assistant as A
from . import team
from .community import api as community_api
from .community import college as community_college
from .community import interests as community_interests
from .community import messages as community_messages
from .community import openings as community_openings
from .community import questions as community_questions
from .community import related as community_related
from .models import ClaimType
from .config import REPO_ROOT, save_env_value, settings
from .llm import BackendError, all_status, get_backend
from .models import Draft, Ledger, Outline, Project, Sentence, now_iso
from .nli import get_scorer
from .sample import demo_ledger

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="PaperSmithAI", version=__version__)

PUBLIC_PATHS = {"/api/health", "/api/community/institutions"}
_PUBLIC_DEPARTMENTS = re.compile(r"^/api/community/institutions/[A-Za-z0-9-]+/departments$")      # the sign-up form
_PROJECT_PATH = re.compile(r"^/api/projects/([A-Za-z0-9]+)")
_JOB_PATH = re.compile(r"^/api/jobs/([^/]+)")
_REVIEWER_WRITES = re.compile(r"^/api/projects/[A-Za-z0-9]+/(reviews(/[A-Za-z0-9]+)?|members/[A-Za-z0-9-]+)$")


@app.middleware("http")
async def signed_in_only(request: Request, call_next):
    """Hosted version: every API call needs a valid sign-in, and a paper is visible only to its owner and the
    people they added (co-authors edit, reviewers read and comment)."""
    path = request.url.path
    if (not auth.enabled() or request.method == "OPTIONS" or not path.startswith("/api/") or path in PUBLIC_PATHS
            or _PUBLIC_DEPARTMENTS.match(path)):
        return await call_next(request)
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    who = await run_in_threadpool(auth.user_for, token)
    if not who:
        return JSONResponse({"detail": "Please sign in again."}, status_code=401)
    m = _PROJECT_PATH.match(path)
    if m:
        role = await run_in_threadpool(storage.access_of, m.group(1), who[0])
        if role is None:
            return JSONResponse({"detail": "project not found"}, status_code=404)
        if role == "reviewer" and request.method not in ("GET", "HEAD") and not _REVIEWER_WRITES.match(path):
            return JSONResponse({"detail": "Reviewers can read and comment, but not change the paper."}, status_code=403)
    m = _JOB_PATH.match(path)
    if m:
        try:
            role = await run_in_threadpool(storage.access_of, jobs.get(m.group(1)).project_id, who[0])
        except KeyError:
            role = None
        if role is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
    reset = auth.current_user.set(who[0])
    reset_email = auth.current_email.set(who[1])
    try:
        return await call_next(request)
    finally:
        auth.current_user.reset(reset)
        auth.current_email.reset(reset_email)


app.include_router(community_api.router)
app.include_router(community_openings.router)
app.include_router(community_questions.router)
app.include_router(community_college.router)
app.include_router(community_related.router)
app.include_router(community_messages.router)
app.include_router(team.router)

try:
    from neo4j.exceptions import DriverError, Neo4jError

    @app.exception_handler(DriverError)
    @app.exception_handler(Neo4jError)
    async def community_unreachable(request: Request, exc: Exception):
        logging.getLogger("papersmith.community").warning("community graph error: %s", exc)
        return JSONResponse({"detail": "The community is unavailable for a moment. Please try again shortly."},
                            status_code=503)
except ImportError:                     # the laptop version keeps the community in a JSON file
    pass


# added last so it runs first: browsers get CORS headers even on a 401
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- helpers

def _load(pid: str) -> Project:
    try:
        return storage.load(pid)
    except (KeyError, ValueError):
        raise HTTPException(404, "project not found") from None


def _edit(pid: str):
    _load(pid)  # 404 early
    return storage.edit(pid)


def _log(project: Project, action: str, **detail) -> None:
    project.history.append({"at": now_iso(), "action": action, **detail})


def _find_sentence(project: Project, sid: str):
    if not project.draft:
        raise HTTPException(404, "no draft")
    for p in project.draft.paragraphs:
        for i, s in enumerate(p.sentences):
            if s.id == sid:
                return p, i, s
    raise HTTPException(404, "sentence not found")


def _find_outline_paragraph(project: Project, para_id: str):
    if not project.outline:
        raise HTTPException(400, "no outline")
    for section in project.outline.sections:
        for p in section.paragraphs:
            if p.id == para_id:
                return section.name, p
    raise HTTPException(404, "paragraph not in outline")


# ---------------------------------------------------------------- status

@app.get("/api/health")
def health():
    scorer = get_scorer()
    config = {
        "nli_model": settings.nli_model,
        "nli_device": settings.nli_device,
        "max_rewrite_attempts": settings.max_rewrite_attempts,
    }
    if not settings.hosted:                 # machine details stay private on the public server
        config.update({"data_dir": str(settings.data_dir), "ollama_url": settings.ollama_url,
                       "ollama_num_ctx": settings.ollama_num_ctx, "env_file": str(REPO_ROOT / ".env")})
    return {
        "version": __version__,
        "hosted": settings.hosted,
        "backends": all_status(),
        "nli": scorer.status() if scorer else {"enabled": False, "loaded": False, "detail": "disabled"},
        "config": config,
    }


class BackendChoice(BaseModel):
    backend: str


@app.put("/api/settings/backend")
def choose_backend(body: BackendChoice):
    """Switch the model PaperSmith uses (takes effect at once, and is saved to .env for the next start)."""
    if settings.hosted:
        raise HTTPException(403, "The model is fixed on the hosted version.")
    name = body.backend.lower()
    if name not in ("ollama", "groq", "claude"):
        raise HTTPException(400, "backend must be ollama, groq or claude")
    status = all_status()[name]
    if not status["available"]:
        raise HTTPException(400, f"{name} is not available: {status['detail']}")
    settings.default_backend = name
    save_env_value("PAPERSMITH_BACKEND", name)
    return health()


@app.post("/api/nli/warmup")
def nli_warmup():
    scorer = get_scorer()
    if not scorer:
        raise HTTPException(400, "NLI disabled")
    scorer.load()
    return scorer.status()


# ---------------------------------------------------------------- projects

class NewProject(BaseModel):
    name: str
    demo: bool = False


def _owner_filter() -> str | None:
    return auth.current_user.get() if auth.enabled() else None


@app.get("/api/projects")
def list_projects():
    return [{k: v for k, v in s.items() if k not in ("owner", "members")} for s in storage.list_all(owner=_owner_filter())]


@app.get("/api/me")
def me():
    return {"signed_in": auth.enabled(), "user": auth.current_user.get()}


@app.post("/api/projects")
def create_project(body: NewProject):
    if auth.enabled() and len(storage.list_all(owner=_owner_filter())) >= settings.max_papers:
        raise HTTPException(429, f"You can keep up to {settings.max_papers} papers. Delete one to start another.")
    ledger = demo_ledger() if body.demo else Ledger(title=body.name)
    project = Project(id=storage.new_id(), name=body.name, ledger=ledger, owner=auth.current_user.get())
    _log(project, "created", demo=body.demo)
    A.welcome(project, mentor=workspace.mentor_on())
    if body.demo:
        workspace._advance(project)        # the demo ledger is complete: go straight to the plan
    return storage.save(project)


_recovered: set[str] = set()


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    project = _load(pid)
    if pid not in _recovered:              # first visit since the server started
        _recovered.add(pid)
        workspace.recover(pid)
        if not project.messages:
            with storage.edit(pid) as p:
                A.welcome(p, mentor=workspace.mentor_on())
                if p.ledger.claims:
                    workspace._advance(p)
        project = _load(pid)
    return project


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    _load(pid)
    storage.delete(pid)
    sources.delete_all(pid)
    return {"ok": True}


# ---------------------------------------------------------------- chat workspace

class ChatIn(BaseModel):
    text: str
    backend: str | None = None
    use_nli: bool = True


@app.post("/api/projects/{pid}/chat")
def post_chat(pid: str, body: ChatIn):
    _load(pid)
    if not body.text.strip():
        raise HTTPException(400, "empty message")
    if not auth.spend("message", settings.daily_messages):
        raise HTTPException(429, f"You've reached today's limit of {settings.daily_messages} messages. It resets tomorrow.")
    uid = auth.current_user.get()
    mid = workspace.post_user_message(pid, body.text, author=uid, author_name=community_api.display_name(uid) if uid else "")
    workspace.start_chat(pid, body.backend, body.use_nli)
    community_interests.touch(pid)          # what the paper is about may have changed
    return {"message_id": mid}


@app.post("/api/projects/{pid}/files")
async def upload_files(pid: str, files: list[UploadFile], backend: str | None = None, use_nli: bool = True):
    _load(pid)
    if not auth.spend("upload", settings.daily_uploads, amount=len(files)):
        raise HTTPException(429, f"You've reached today's limit of {settings.daily_uploads} files. It resets tomorrow.")
    payload = [(f.filename or "file", await f.read()) for f in files]
    accepted, rejected = workspace.add_files(pid, payload, uploaded_by=auth.current_user.get())
    for fid in accepted:
        workspace.start_ingest(pid, fid, backend, use_nli)
    if accepted:
        community_interests.touch(pid, delay=240)      # after the files have been read
    return {"accepted": accepted, "rejected": rejected}


class RoleIn(BaseModel):
    role: str


@app.patch("/api/projects/{pid}/files/{fid}")
def set_file_role(pid: str, fid: str, body: RoleIn, backend: str | None = None, use_nli: bool = True):
    if body.role not in ("own", "reference", "data"):
        raise HTTPException(400, "role must be own, reference or data")
    _load(pid)
    try:
        workspace.set_role(pid, fid, body.role)
    except KeyError:
        raise HTTPException(404, "file not found") from None
    workspace.start_ingest(pid, fid, backend, use_nli)
    return _load(pid)


@app.post("/api/projects/{pid}/files/{fid}/retry")
def retry_file(pid: str, fid: str, backend: str | None = None, use_nli: bool = True):
    with _edit(pid) as p:
        f = next((x for x in p.sources if x.id == fid), None)
        if not f:
            raise HTTPException(404, "file not found")
        f.status, f.detail = "queued", "trying again"
    workspace.start_ingest(pid, fid, backend, use_nli)
    return _load(pid)


@app.delete("/api/projects/{pid}/files/{fid}")
def remove_file(pid: str, fid: str):
    _load(pid)
    try:
        workspace.delete_file(pid, fid)
    except KeyError:
        raise HTTPException(404, "file not found") from None
    return _load(pid)


@app.get("/api/projects/{pid}/files/{fid}/passages")
def file_passages(pid: str, fid: str):
    _load(pid)
    return [p for p in sources.load(pid) if p.source_id == fid]


@app.get("/api/projects/{pid}/claims/{cid}/evidence")
def claim_evidence(pid: str, cid: str):
    project = _load(pid)
    claim = project.ledger.claim_map().get(cid)
    if not claim:
        raise HTTPException(404, "claim not found")
    passages = sources.by_id(pid)
    files = {f.id: f.filename for f in project.sources}
    out = []
    for sid in claim.sources:
        ps = passages.get(sid)
        if ps:
            label = {"chat": "Chat", "suggestion": "Accepted suggestion"}.get(ps.source_id) or files.get(ps.source_id, ps.source_id)
            out.append({"id": ps.id, "file": label,
                        "location": ps.location, "text": ps.text})
    return {"claim": claim, "evidence": out}


class ClaimPatch(BaseModel):
    text: str | None = None
    type: str | None = None


@app.patch("/api/projects/{pid}/claims/{cid}")
def edit_claim(pid: str, cid: str, body: ClaimPatch):
    with _edit(pid) as p:
        claim = p.ledger.claim_map().get(cid)
        if not claim:
            raise HTTPException(404, "claim not found")
        before = claim.text
        if body.text is not None and body.text.strip():
            claim.text = body.text.strip()
            claim.origin = "manual" if claim.origin == "manual" else claim.origin
        retyped = False
        if body.type:
            try:
                new_type = ClaimType(body.type)
            except ValueError:
                raise HTTPException(400, "unknown claim type") from None
            retyped = new_type != claim.type
            claim.type = new_type
        if p.outline and retyped:
            # a retyped claim belongs to another section: re-plan before anything is written, otherwise move just it
            if p.draft is None:
                p.outline = planner.plan(p.ledger, p.settings.target_pages)
            else:
                planner.remove_claim(p.outline, cid)
                planner.place_claim(p.outline, p.ledger, claim)
                planner.refresh_abstract(p.outline, p.ledger)
        if p.outline:
            for s in p.outline.sections:
                for para in s.paragraphs:
                    if cid in para.claim_ids:
                        para.stale = True
        _log(p, "claim_edited", claim=cid, before=before, after=claim.text)
    return p


@app.delete("/api/projects/{pid}/claims/{cid}")
def remove_claim(pid: str, cid: str):
    with _edit(pid) as p:
        if cid not in p.ledger.claim_map():
            raise HTTPException(404, "claim not found")
        p.ledger.claims = [c for c in p.ledger.claims if c.id != cid]
        if p.outline:
            planner.remove_claim(p.outline, cid)
        _log(p, "claim_removed", claim=cid)
    return p


class WriteIn(BaseModel):
    section: str | None = None
    backend: str | None = None
    use_nli: bool = True


@app.post("/api/projects/{pid}/write")
def write_now(pid: str, body: WriteIn):
    _load(pid)
    job = workspace.start_write(pid, body.backend, body.use_nli, section=body.section)
    return {"started": job is not None}


class ProjectPatch(BaseModel):
    name: str


@app.patch("/api/projects/{pid}")
def rename_project(pid: str, body: ProjectPatch):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name cannot be empty")
    with _edit(pid) as project:
        _log(project, "renamed", before=project.name, after=name)
        project.name = name
    return project


@app.post("/api/projects/{pid}/duplicate")
def duplicate_project(pid: str):
    source = _load(pid)
    copy = source.model_copy(deep=True)
    copy.id = storage.new_id()
    copy.name = f"{source.name} (copy)"
    copy.created_at = now_iso()
    copy.owner = auth.current_user.get() or source.owner
    _log(copy, "duplicated", source=pid)
    for rel in storage.project_files(source):
        path = storage.local_file(pid, rel)
        if path.is_file():
            storage.write_file(copy.id, rel, path.read_bytes())
    sources.save(copy.id, sources.load(pid))
    return storage.save(copy)


# ---------------------------------------------------------------- figure files

FIGURE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".pdf": "application/pdf",
                ".svg": "image/svg+xml", ".eps": "application/postscript"}
MAX_FIGURE_BYTES = 20 * 1024 * 1024


@app.post("/api/projects/{pid}/figures/{fid}/file")
async def upload_figure(pid: str, fid: str, file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in FIGURE_TYPES:
        raise HTTPException(400, f"figures must be one of {', '.join(FIGURE_TYPES)}")
    data = await file.read()
    if len(data) > MAX_FIGURE_BYTES:
        raise HTTPException(400, "figure files are limited to 20 MB")
    with _edit(pid) as project:
        figure = next((f for f in project.ledger.figures if f.id == fid), None)
        if not figure:
            raise HTTPException(404, "figure not found")
        if figure.filename:
            storage.remove_file(pid, figure.filename)
        figure.filename = f"{fid.lower()}{ext}"
        storage.write_file(pid, figure.filename, data)
        _log(project, "figure_uploaded", figure=fid, bytes=len(data))
    return project


@app.get("/api/projects/{pid}/files/{name}")
def get_file(pid: str, name: str):
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(404, "file not found")
    try:
        target = storage.local_file(pid, name)
    except ValueError:
        raise HTTPException(404, "file not found") from None
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target, media_type=FIGURE_TYPES.get(target.suffix.lower(), "application/octet-stream"))


@app.put("/api/projects/{pid}/ledger")
def update_ledger(pid: str, ledger: Ledger):
    ids = [c.id for c in ledger.claims]
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "claim IDs must be unique")
    with _edit(pid) as project:
        project.ledger = ledger
        _log(project, "ledger_saved", claims=len(ledger.claims), references=len(ledger.references))
    return project


class BibImport(BaseModel):
    text: str
    replace: bool = False


@app.post("/api/projects/{pid}/bibtex")
def import_bibtex(pid: str, body: BibImport):
    refs = bibtex.parse(body.text)
    if not refs:
        raise HTTPException(400, "no BibTeX entries found")
    with _edit(pid) as project:
        existing = {} if body.replace else project.ledger.ref_map()
        for r in refs:
            existing[r.key] = r
        project.ledger.references = list(existing.values())
        _log(project, "bibtex_imported", entries=len(refs))
    return project


# ---------------------------------------------------------------- outline

@app.post("/api/projects/{pid}/outline/plan")
def plan_outline(pid: str):
    with _edit(pid) as project:
        if not project.ledger.claims:
            raise HTTPException(400, "add claims to the ledger first")
        project.outline = planner.plan(project.ledger)
        _log(project, "outline_planned")
    return {"outline": project.outline, "problems": planner.validate(project.outline, project.ledger)}


@app.put("/api/projects/{pid}/outline")
def save_outline(pid: str, outline: Outline):
    with _edit(pid) as project:
        project.outline = outline
        _log(project, "outline_saved", approved=outline.approved)
    return {"outline": outline, "problems": planner.validate(outline, project.ledger)}


# ---------------------------------------------------------------- drafting

class DraftRequest(BaseModel):
    backend: str | None = None
    model: str | None = None
    use_nli: bool = True


@app.post("/api/projects/{pid}/draft")
def start_draft(pid: str, body: DraftRequest):
    project = _load(pid)
    if not project.ledger.claims:
        raise HTTPException(400, "add claims to the ledger first")
    try:
        backend = get_backend(body.backend, body.model)
    except BackendError as exc:
        raise HTTPException(400, str(exc)) from None
    if project.outline is None:
        with _edit(pid) as p:
            p.outline = planner.plan(p.ledger)
            project = p

    outline = project.outline
    ledger = project.ledger

    def run(job: jobs.Job) -> None:
        job.total = sum(len(s.paragraphs) for s in outline.sections)
        job.message = f"Writing with {backend.name} ({backend.model})"
        with storage.edit(pid) as p:
            p.draft = None
            _log(p, "draft_started", backend=backend.name, model=backend.model)

        def on_paragraph(para, i, total):
            job.progress = i
            job.message = f"Wrote {para.section} paragraph {para.id} ({i}/{total})"
            with storage.edit(pid) as p:
                if p.draft is None:
                    p.draft = Draft(backend=backend.name, model=backend.model)
                p.draft.paragraphs.append(para)
                p.draft.updated_at = now_iso()

        writer.write_draft(backend, ledger, outline, on_paragraph=on_paragraph, use_nli=body.use_nli)
        with storage.edit(pid) as p:
            p.consistency = consistency.check_all(p.ledger, p.draft, use_nli=body.use_nli)
            _log(p, "draft_finished", **{k: v for k, v in provenance.stats(p).items()
                                         if k in ("sentences", "flagged", "leakage_rate_pct")})
        job.message = "Done"

    try:
        job = jobs.start(pid, "draft", run)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return job.to_dict()


@app.post("/api/projects/{pid}/paragraphs/{para_id}/regenerate")
def regenerate_paragraph(pid: str, para_id: str, body: DraftRequest):
    project = _load(pid)
    section, outline_para = _find_outline_paragraph(project, para_id)
    try:
        backend = get_backend(body.backend, body.model)
    except BackendError as exc:
        raise HTTPException(400, str(exc)) from None

    def run(job: jobs.Job) -> None:
        job.total = 1
        job.message = f"Rewriting {para_id}"
        para = writer.write_paragraph(backend, project.ledger, section, outline_para, use_nli=body.use_nli)
        with storage.edit(pid) as p:
            if p.draft is None:
                p.draft = Draft(backend=backend.name, model=backend.model)
            replaced = False
            for i, existing in enumerate(p.draft.paragraphs):
                if existing.id == para_id:
                    p.draft.paragraphs[i] = para
                    replaced = True
            if not replaced:
                p.draft.paragraphs.append(para)
            _log(p, "paragraph_regenerated", paragraph=para_id, backend=backend.name)
        job.progress = 1
        job.message = "Done"

    try:
        job = jobs.start(pid, "regenerate", run)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return job.to_dict()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return jobs.get(job_id).to_dict()
    except KeyError:
        raise HTTPException(404, "job not found") from None


@app.get("/api/projects/{pid}/jobs")
def project_jobs(pid: str):
    return [j.to_dict() for j in jobs.for_project(pid)]


# ---------------------------------------------------------------- human review

class SentenceEdit(BaseModel):
    text: str
    claim_ids: list[str] | None = None
    ref_keys: list[str] | None = None


@app.put("/api/projects/{pid}/sentences/{sid}")
def edit_sentence(pid: str, sid: str, body: SentenceEdit):
    with _edit(pid) as project:
        para, _, s = _find_sentence(project, sid)
        old = s.text
        s.text = body.text.strip()
        if body.claim_ids is not None:
            s.claim_ids = body.claim_ids
            s.signpost = not body.claim_ids
        if body.ref_keys is not None:
            s.ref_keys = body.ref_keys
        s.status = "user_edited"
        verifier.verify_paragraph(para, project.ledger)
        _log(project, "sentence_edited", sentence=sid, before=old, after=s.text)
    return para


class NewSentence(BaseModel):
    text: str
    claim_ids: list[str] = []
    ref_keys: list[str] = []
    after: str | None = None


@app.post("/api/projects/{pid}/paragraphs/{para_id}/sentences")
def add_sentence(pid: str, para_id: str, body: NewSentence):
    with _edit(pid) as project:
        if not project.draft:
            raise HTTPException(404, "no draft")
        para = next((p for p in project.draft.paragraphs if p.id == para_id), None)
        if not para:
            raise HTTPException(404, "paragraph not found")
        n = 1 + max((int(s.id.split(".S")[-1]) for s in para.sentences if ".S" in s.id), default=0)
        s = Sentence(id=f"{para_id}.S{n}", text=body.text.strip(), claim_ids=body.claim_ids, ref_keys=body.ref_keys,
                     signpost=not body.claim_ids, origin="human", status="user_edited")
        idx = next((i + 1 for i, x in enumerate(para.sentences) if x.id == body.after), len(para.sentences))
        para.sentences.insert(idx, s)
        verifier.verify_paragraph(para, project.ledger)
        _log(project, "sentence_added", sentence=s.id, text=s.text)
    return para


@app.post("/api/projects/{pid}/sentences/{sid}/accept")
def accept_sentence(pid: str, sid: str):
    with _edit(pid) as project:
        para, _, s = _find_sentence(project, sid)
        s.status = "accepted"
        _log(project, "sentence_accepted", sentence=sid, flags=[f.kind for f in s.flags])
    return para


@app.delete("/api/projects/{pid}/sentences/{sid}")
def delete_sentence(pid: str, sid: str):
    with _edit(pid) as project:
        para, i, s = _find_sentence(project, sid)
        para.sentences.pop(i)
        verifier.verify_paragraph(para, project.ledger)
        _log(project, "sentence_deleted", sentence=sid, text=s.text)
    return para


@app.post("/api/projects/{pid}/verify")
def reverify(pid: str, use_nli: bool = True):
    with _edit(pid) as project:
        if project.draft:
            for p in project.draft.paragraphs:
                verifier.verify_paragraph(p, project.ledger, use_nli)
        project.consistency = consistency.check_all(project.ledger, project.draft, use_nli)
        _log(project, "reverified")
    return project


@app.post("/api/projects/{pid}/consistency")
def run_consistency(pid: str, use_nli: bool = True):
    with _edit(pid) as project:
        project.consistency = consistency.check_all(project.ledger, project.draft, use_nli)
    return project.consistency


@app.get("/api/projects/{pid}/stats")
def project_stats(pid: str):
    project = _load(pid)
    return {"stats": provenance.stats(project), "disclosure": provenance.disclosure_statement(project)}


# ---------------------------------------------------------------- export

@app.get("/api/projects/{pid}/export/{fmt}")
def export_project(pid: str, fmt: str):
    project = _load(pid)
    words = re.findall(r"[A-Za-z0-9]+", project.ledger.title or project.name)
    slug = "_".join(words)[:60].strip("_") or "paper"
    if fmt == "latex":
        return PlainTextResponse(export.to_latex(project), headers={"Content-Disposition": f'attachment; filename="{slug}.tex"'})
    if fmt == "bib":
        return PlainTextResponse(export.to_bib(project), headers={"Content-Disposition": 'attachment; filename="references.bib"'})
    if fmt == "markdown":
        return PlainTextResponse(export.to_markdown(project), headers={"Content-Disposition": f'attachment; filename="{slug}.md"'})
    if fmt == "disclosure":
        return PlainTextResponse(provenance.disclosure_markdown(project),
                                 headers={"Content-Disposition": f'attachment; filename="{slug}_ai_disclosure.md"'})
    if fmt == "docx":
        return Response(export.to_docx(project),
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        headers={"Content-Disposition": f'attachment; filename="{slug}.docx"'})
    if fmt == "zip":
        return Response(export.bundle(project), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{slug}.zip"'})
    raise HTTPException(404, "unknown format")


# ---------------------------------------------------------------- frontend (production build)

_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        target = _dist / path
        if path and target.is_file() and _dist in target.resolve().parents:
            return FileResponse(target)
        # index.html must never be cached, or a rebuilt site keeps loading the previous bundle
        return FileResponse(Path(_dist) / "index.html", headers={"Cache-Control": "no-cache"})
