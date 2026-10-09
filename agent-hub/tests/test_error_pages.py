"""Browsers get the designed 404 and 500 pages; the API keeps answering in JSON."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent_hub import main

PAGE = {"accept": "text/html,application/xhtml+xml"}


def test_an_unknown_page_shows_the_404_page_with_status_404():
    with TestClient(main.app) as client:
        out = client.get("/no/such/page", headers=PAGE)
    assert out.status_code == 404 and "text/html" in out.headers["content-type"]
    assert "Page not found" in out.text and out.headers["x-frame-options"] == "DENY"


def test_the_api_keeps_json_for_unknown_paths_and_non_browser_requests():
    with TestClient(main.app) as client:
        api = client.get("/api/no-such-route", headers=PAGE)
        plain = client.get("/no/such/page")
    assert api.status_code == 404 and api.json() == {"detail": "Not Found"}
    assert plain.status_code == 404 and plain.json() == {"detail": "Not Found"}


def test_a_crash_on_a_page_shows_the_500_page_and_on_the_api_json():
    @main.app.get("/test-crash-page")
    async def boom():  # noqa: ANN202
        raise RuntimeError("boom")

    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            page = client.get("/test-crash-page", headers=PAGE)
            api = client.get("/test-crash-page")
        assert page.status_code == 500 and "Something went wrong on our side" in page.text
        assert page.headers["x-content-type-options"] == "nosniff"
        assert api.status_code == 500 and api.json() == {"detail": "Something went wrong on the hub."}
    finally:
        main.app.router.routes[:] = [r for r in main.app.router.routes if getattr(r, "path", "") != "/test-crash-page"]
