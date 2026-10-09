"""Content-Security-Policy for the pages this app serves.

Each page is one HTML file with its script inline. The policy allows exactly those inline scripts (by the SHA-256 of
their text, read from the file as served, so it can't go stale when a page changes), styles and fonts from this app and
Google Fonts, and requests only to this app. Anything injected into a page (a script tag, an event handler, a call to
another site) is refused by the browser. The same file is used by all three apps."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from fastapi.responses import FileResponse

_INLINE_SCRIPT = re.compile(rb"<script(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>", re.S | re.I)
_cache: dict[str, tuple[int, int, str]] = {}  # page path -> (mtime, size, policy)

PERMISSIONS_POLICY = "camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), bluetooth=()"


def _lf(text: bytes) -> bytes:
    """Browsers hash a script's text after turning every CRLF or lone CR into LF (HTML input preprocessing)."""
    return text.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _sha(text: bytes) -> str:
    return f"'sha256-{base64.b64encode(hashlib.sha256(_lf(text)).digest()).decode()}'"


def csp_for(path: Path) -> str:
    stat = path.stat()
    cached = _cache.get(str(path))
    if cached is None or cached[:2] != (stat.st_mtime_ns, stat.st_size):
        hashes = " ".join(_sha(body) for body in _INLINE_SCRIPT.findall(path.read_bytes()))
        policy = "; ".join([
            "default-src 'self'",
            f"script-src 'self' {hashes}".strip(),
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com data:",
            "img-src 'self' data: blob:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'none'",
            "form-action 'self'",
            "object-src 'none'",
        ])
        cached = _cache[str(path)] = (stat.st_mtime_ns, stat.st_size, policy)
    return cached[2]


def page(path: Path, status_code: int = 200, headers: dict[str, str] | None = None) -> FileResponse:
    """Serve a page with its Content-Security-Policy."""
    return FileResponse(path, status_code=status_code, headers={**(headers or {}), "Content-Security-Policy": csp_for(path)})
