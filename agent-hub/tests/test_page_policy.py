"""Every page is served with a Content-Security-Policy that allows exactly its own inline scripts (hashed the way
browsers hash them) and a Permissions-Policy; API responses carry the Permissions-Policy too."""

from __future__ import annotations

import base64
import hashlib

from fastapi.testclient import TestClient

from agent_hub import main, pagepolicy

PAGE = {"accept": "text/html"}


def sha(text: bytes) -> str:
    return f"'sha256-{base64.b64encode(hashlib.sha256(text).digest()).decode()}'"


def test_inline_scripts_are_hashed_as_browsers_see_them(tmp_path):
    page = tmp_path / "page.html"
    page.write_bytes(b"<html><script>\r\nconsole.log(1);\r\n</script><script src=\"/x.js\"></script><script type=\"module\">a()</script></html>")
    policy = pagepolicy.csp_for(page)
    assert sha(b"\nconsole.log(1);\n") in policy and sha(b"a()") in policy  # CRLF counted as LF; the src= script not hashed
    assert "'unsafe-inline'" not in policy.split("script-src")[1].split(";")[0]
    assert "frame-ancestors 'none'" in policy and "object-src 'none'" in policy and "connect-src 'self'" in policy
    page.write_bytes(b"<script>b()</script>")
    assert sha(b"b()") in pagepolicy.csp_for(page)  # a changed page gets a fresh policy


def test_the_pages_carry_their_policy_and_the_api_its_permissions(engine):
    main.app.state.engine = engine
    try:
        with TestClient(main.app) as client:
            for path in ("/", "/admin"):
                out = client.get(path, headers=PAGE)
                assert "script-src 'self' 'sha256-" in out.headers["content-security-policy"], path
                assert out.headers["permissions-policy"] == pagepolicy.PERMISSIONS_POLICY
            missing = client.get("/nowhere", headers=PAGE)
            assert missing.status_code == 404 and "content-security-policy" in missing.headers
            assert client.get("/api/agents").headers["permissions-policy"].startswith("camera=()")
    finally:
        main.app.state.engine = None


def test_privacy_page_favicon_and_robots(engine):
    main.app.state.engine = engine
    try:
        with TestClient(main.app) as client:
            privacy = client.get("/privacy", headers=PAGE)
            assert privacy.status_code == 200 and "Your data, plainly." in privacy.text
            assert "script-src 'self' 'sha256-" in privacy.headers["content-security-policy"]
            assert "Hindsight Cloud" in privacy.text and "Gemini" in privacy.text
            robots = client.get("/robots.txt")
            assert robots.status_code == 200 and robots.text == "User-agent: *\nDisallow: /\n"
            icon = client.get("/favicon.svg")
            assert icon.status_code == 200 and icon.headers["content-type"].startswith("image/svg+xml")
            old = client.get("/favicon.ico", follow_redirects=False)
            assert old.status_code == 308 and old.headers["location"] == "/favicon.svg"
            assert 'href="/privacy"' in client.get("/", headers=PAGE).text
    finally:
        main.app.state.engine = None
