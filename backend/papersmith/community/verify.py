"""Proving you belong to a college: a one-time code sent to your college email.

Sign-in email is not proof (anyone can sign up with any address), so membership is verified separately.
A verified domain belongs to the college: a listed college already knows its website's domain (from the
directory), an unlisted one learns it from its first verified member; later members must use one of its
domains (or a subdomain, e.g. student.nitt.edu), and a domain another college owns moves the person to that
college, which is how duplicate college names get merged.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import smtplib
import threading
import time
from email.message import EmailMessage

from ..config import settings
from . import core, directory
from .store import store

FREE_MAIL = {"gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "ymail.com", "outlook.com", "hotmail.com",
             "live.com", "msn.com", "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com", "zoho.com",
             "rediffmail.com", "gmx.com", "mail.com", "yandex.com", "tutanota.com"}
ACADEMIC = re.compile(r"(^|\.)(edu|ac)(\.[a-z]{2})?$|\.edu\.[a-z]{2}$|\.ac\.[a-z]{2}$")
CODE_TTL = 15 * 60
MAX_ATTEMPTS = 5
MAX_SENDS_PER_HOUR = 3

_pending: dict[str, dict] = {}
_lock = threading.Lock()


def available() -> bool:
    return bool(settings.smtp_user and settings.smtp_password)


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def domain_of(email: str) -> str:
    m = re.fullmatch(r"[^@\s]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", email.strip())
    if not m:
        raise ValueError("That doesn't look like an email address.")
    return m.group(1).lower()


def _college_check(uid: str, domain: str) -> tuple[dict | None, dict | None]:
    """(college the person claims, college that owns this domain). Raises ValueError when the domain can't
    prove membership."""
    if domain in FREE_MAIL:
        raise ValueError("Use your college email, not a personal one.")
    s = store()
    me = s.get_person(uid) or {}
    mine = s.institution(me.get("institution", "")) if me.get("institution") else None
    owner = s.institution_for_domain(domain)
    if owner is None:
        listed = directory.for_domain(domain)
        if listed:
            owner = core.college_from_directory(listed)
    if mine and not mine["domains"] and mine.get("website"):
        mine = {**mine, "domains": [mine["website"]]}
    if owner is None and mine and mine["domains"] and not ACADEMIC.search(domain):
        raise ValueError(f"That email isn't from {mine['name']} ({', '.join(mine['domains'])}).")
    if owner is None and mine is None:
        raise ValueError("Choose your college first.")
    return mine, owner


def start(uid: str, email: str) -> str:
    if not available():
        raise ValueError("College verification isn't switched on yet.")
    domain = domain_of(email)
    mine, owner = _college_check(uid, domain)
    now = time.time()
    with _lock:
        entry = _pending.get(uid, {})
        sends = [t for t in entry.get("sends", []) if now - t < 3600]
        if len(sends) >= MAX_SENDS_PER_HOUR:
            raise ValueError("Too many codes requested. Try again in an hour.")
        code = f"{secrets.randbelow(10**6):06d}"
        _pending[uid] = {"email": email.strip().lower(), "domain": domain, "hash": _hash(code),
                         "expires": now + CODE_TTL, "attempts": 0, "sends": sends + [now]}
    college = (owner or mine or {}).get("name", "your college")
    _send(email.strip(), f"Your PaperSmith code: {code}",
          f"Your code to verify your {college} email on PaperSmith is {code}.\n\n"
          f"It expires in 15 minutes. If you didn't ask for it, you can ignore this email.")
    return college


def confirm(uid: str, code: str) -> dict:
    with _lock:
        entry = _pending.get(uid)
        if not entry or entry["expires"] < time.time():
            raise ValueError("The code expired. Ask for a new one.")
        entry["attempts"] += 1
        if entry["attempts"] > MAX_ATTEMPTS:
            _pending.pop(uid, None)
            raise ValueError("Too many wrong codes. Ask for a new one.")
        if not hmac.compare_digest(entry["hash"], _hash(code.strip())):
            raise ValueError("That code isn't right.")
        _pending.pop(uid, None)
    s = store()
    domain = entry["domain"]
    mine, owner = _college_check(uid, domain)
    if owner:
        college = owner                     # the domain already belongs to a college: join that one
    else:
        college = s.upsert_institution(mine["name"], domain=domain, key=mine["key"])
    if domain not in college.get("domains", []):
        s.upsert_institution(college["name"], domain=domain, key=college["key"])
    return s.upsert_person(uid, institution=college["key"], verified=True, email_domain=domain)


def _send(to: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = settings.smtp_from or f"PaperSmith <{settings.smtp_user}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        raise ValueError("Couldn't send the email right now. Try again in a few minutes.") from exc
