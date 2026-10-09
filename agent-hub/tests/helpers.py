"""Small helpers for driving the engine in tests."""

from __future__ import annotations

from agent_hub.engine import Engine

PDF = ("Cedarline thread.eml", b"From: a\nSubject: renewal\n\nWe need the data residency clause.")
RFP_DOC = ("Acme Security Questionnaire.docx", b"fake docx bytes")


async def say(eng: Engine, conv: str, text: str = "", files: list[tuple[str, bytes]] | tuple = ()) -> list[dict]:
    """Send a message, wait until everything it started has finished, return the new events."""
    before = eng.store.last_n(conv)
    await eng.handle_message(conv, text, list(files))
    await eng.wait_idle(conv)
    return eng.store.events(conv, before)


async def click(eng: Engine, conv: str, action_id: str) -> list[dict]:
    before = eng.store.last_n(conv)
    await eng.handle_action(conv, action_id)
    await eng.wait_idle(conv)
    return eng.store.events(conv, before)


def assistant(events: list[dict]) -> list[dict]:
    return [e for e in events if e["role"] == "assistant"]


def texts(events: list[dict]) -> str:
    return "\n".join(e.get("text") or "" for e in events if e["role"] == "assistant")


def all_text(events: list[dict]) -> str:
    """Every visible string in the events (messages, card text, bullets, rows, links)."""
    out: list[str] = []

    def walk(value) -> None:  # noqa: ANN001
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in ("n", "created_at"):
                    walk(v)

    walk(events)
    return "\n".join(out)


def actions_of(events: list[dict]) -> list[dict]:
    for e in reversed(events):
        if e.get("actions"):
            return e["actions"]
    return []


def action(events: list[dict], label_part: str) -> dict:
    for a in actions_of(events):
        if label_part.lower() in a["label"].lower():
            return a
    raise AssertionError(f"no action containing {label_part!r} in {[a['label'] for a in actions_of(events)]}")


def cards(events: list[dict]) -> list[dict]:
    return [c for e in events for c in e.get("cards") or []]


def links(events: list[dict]) -> list[dict]:
    return [link for c in cards(events) for link in c.get("links") or []]
