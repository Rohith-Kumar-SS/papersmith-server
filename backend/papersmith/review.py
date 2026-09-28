"""Which sentences still need the author's attention (mirrors frontend/src/review.ts)."""

from __future__ import annotations

from .models import Sentence


def needs_review(s: Sentence) -> bool:
    return s.status != "accepted" and any(f.severity in ("error", "warning") for f in s.flags)
