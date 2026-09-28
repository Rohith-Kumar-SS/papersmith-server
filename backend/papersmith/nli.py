"""Natural-language-inference scorer (does the source entail the sentence?).

Loaded lazily; if torch/transformers or the model are unavailable the verifier
falls back to its lexical checks only.
"""

from __future__ import annotations

import logging
import threading

from .config import settings

log = logging.getLogger(__name__)


def _pick_device(torch) -> str:
    """NLI_DEVICE=cpu|cuda forces a device. 'auto' uses the GPU only when no local Ollama server
    shares it: on a 6 GB card the writer model needs all the VRAM, and DeBERTa-base is fast enough on CPU."""
    if settings.nli_device == "auto" and settings.default_backend.lower() != "ollama" and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()     # an API model does the writing: the GPU is the checker's
        return "cuda" if free > 1.5 * 1024**3 else "cpu"
    if settings.nli_device in ("cpu", "cuda"):
        return settings.nli_device if settings.nli_device == "cpu" or torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        import httpx

        httpx.get(f"{settings.ollama_url}/api/version", timeout=1)
        return "cpu"
    except Exception:  # noqa: BLE001 - Ollama not running: the GPU is free
        free, _total = torch.cuda.mem_get_info()
        return "cuda" if free > 1.5 * 1024**3 else "cpu"


class NLIScorer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._label_idx: dict[str, int] = {}
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self.error:
                return False
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._device = _pick_device(torch)
                if self._device == "cuda":
                    model = model.half()
                self._model = model.to(self._device).eval()
                labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
                self._label_idx = {
                    "entailment": labels.get("entailment", 0),
                    "contradiction": labels.get("contradiction", 2),
                }
                log.info("NLI model %s loaded on %s", self.model_name, self._device)
                return True
            except Exception as exc:  # noqa: BLE001 - any failure means "no NLI"
                self.error = f"{type(exc).__name__}: {exc}"
                log.warning("NLI unavailable: %s", self.error)
                return False

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[tuple[float, float]]:
        """Return (P(entailment), P(contradiction)) for each (premise, hypothesis)."""
        if not pairs or not self.load():
            return []
        import torch

        out: list[tuple[float, float]] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = self._tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                truncation="only_first", max_length=512, padding=True, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1).cpu()
            e, c = self._label_idx["entailment"], self._label_idx["contradiction"]
            out.extend((float(row[e]), float(row[c])) for row in probs)
        return out

    def status(self) -> dict:
        if not settings.nli_enabled:
            return {"enabled": False, "loaded": False, "detail": "disabled (PAPERSMITH_NLI=0)"}
        if self.error:
            return {"enabled": True, "loaded": False, "detail": self.error}
        return {
            "enabled": True,
            "loaded": self.loaded,
            "detail": f"{self.model_name} on {self._device}" if self.loaded else f"{self.model_name} (loads on first use)",
        }


class LLMScorer:
    """Entailment judged by an API model, for servers too small for the local NLI model (the free Render
    tier has 512 MB). Same interface as NLIScorer; labels become probabilities the checks already use."""

    remote = True
    BATCH = 12
    SYSTEM = (
        "You check whether statements follow from source text. For each numbered pair, label the hypothesis:\n"
        "- entail: everything it says is stated in, or directly follows from, the premise. Paraphrase, reordering, "
        "abbreviations and obvious restatements count as entail.\n"
        "- contradict: the premise says otherwise, including different numbers.\n"
        "- neutral: it adds anything the premise does not state (a fact, number, cause, comparison, evaluation or "
        "implication) or cannot be checked against it.\n"
        "Be strict about added content and lenient about wording. Return JSON."
    )
    SCHEMA = {"type": "object", "required": ["labels"], "properties": {"labels": {"type": "array", "items": {
        "type": "object", "required": ["n", "label"],
        "properties": {"n": {"type": "integer"}, "label": {"type": "string", "enum": ["entail", "neutral", "contradict"]}}}}}}
    PROBS = {"entail": (0.92, 0.02), "neutral": (0.25, 0.05), "contradict": (0.05, 0.9)}
    UNKNOWN = (0.5, 0.0)            # the judge could not be asked: leave the decision to the other checks

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> bool:
        return True

    def _backend(self):
        from .llm import OpenAICompatBackend

        return OpenAICompatBackend(model=settings.groq_read_model, fallback=settings.groq_fallback_model)

    def score(self, pairs: list[tuple[str, str]], batch_size: int = BATCH) -> list[tuple[float, float]]:
        from .llm import BackendError

        out: list[tuple[float, float] | None] = []
        todo: list[int] = []
        for i, (p, h) in enumerate(pairs):
            hit = self._cache.get((p[:1500], h))
            out.append(hit)
            if hit is None:
                todo.append(i)
        for start in range(0, len(todo), batch_size):
            idx = todo[start:start + batch_size]
            lines = []
            for n, i in enumerate(idx, 1):
                p, h = pairs[i]
                lines.append(f"{n}. PREMISE: {' '.join(p.split())[:1500]}\n   HYPOTHESIS: {' '.join(h.split())}")
            try:
                raw = self._backend().generate_json(self.SYSTEM, "\n\n".join(lines), self.SCHEMA, max_tokens=1500, effort="low")
                labels = {int(x.get("n", 0)): str(x.get("label", "")) for x in raw.get("labels", []) if isinstance(x, dict)}
            except BackendError as exc:
                log.info("LLM entailment judge unavailable: %s", exc)
                labels = {}
            for n, i in enumerate(idx, 1):
                probs = self.PROBS.get(labels.get(n, ""), self.UNKNOWN)
                out[i] = probs
                if labels.get(n):
                    with self._lock:
                        if len(self._cache) > 20000:
                            self._cache.clear()
                        self._cache[(pairs[i][0][:1500], pairs[i][1])] = probs
        return [x if x is not None else self.UNKNOWN for x in out]

    def status(self) -> dict:
        return {"enabled": True, "loaded": True, "detail": f"model judge ({settings.groq_read_model})"}


_scorer: NLIScorer | LLMScorer | None = None


def get_scorer() -> NLIScorer | LLMScorer | None:
    global _scorer
    if not settings.nli_enabled:
        return None
    if _scorer is None:
        _scorer = LLMScorer() if settings.nli_backend == "llm" else NLIScorer(settings.nli_model)
    return _scorer
