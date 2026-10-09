"""An administrator asking the chat who uses the platform and how much ("how many companies are working here, how many tokens
did they use, how much storage"). Answered from records with no model call. Anyone who is not an administrator is told so."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import usage

if TYPE_CHECKING:
    from ..engine import Engine


def _n(value: int) -> str:
    return f"{int(value):,}"


def _tokens(row: dict[str, Any]) -> str:
    return f"{_n(row['input_tokens'])} in / {_n(row['output_tokens'])} out"


def card(report: dict[str, Any]) -> dict[str, Any]:
    t = report["totals"]
    sections: list[dict[str, Any]] = [{"heading": "Everyone", "rows": [
        ["Companies", _n(t["companies"])], ["People", _n(t["people"])], ["Chats / messages", f"{_n(t['chats'])} / {_n(t['messages'])}"],
        ["Model calls", f"{_n(t['calls'])}" + (f" ({_n(t['failed'])} failed)" if t["failed"] else "")], ["Tokens", _tokens(t)],
        ["Storage", usage.human_bytes(t["storage_bytes"])]]}]
    for c in report["companies"]:
        parts = ", ".join(f"{name} {_n(c['agents'][a]['calls'])}" for a, name in usage.AGENTS)
        sections.append({"heading": c["name"], "rows": [
            ["People / chats / messages", f"{_n(c['people'])} / {_n(c['chats'])} / {_n(c['messages'])}"],
            ["Model calls", f"{_n(c['calls'])} (hub {_n(c['hub']['calls'])}, {parts})"], ["Tokens", _tokens(c)],
            ["Storage", usage.human_bytes(c["storage_bytes"])]]})
    u = report["unassigned"]
    if u["workspaces"]:
        sections.append({"heading": "Workspaces no company has been given", "text": ", ".join(f"{w['name']} ({w['agent']})" for w in u["workspaces"]),
                         "rows": [["Model calls", _n(u["calls"])], ["Tokens", _tokens(u)], ["Storage", usage.human_bytes(u["storage_bytes"])]]})
    notes = []
    if report["tracking_since"]:
        notes.append(f"Model calls and tokens are counted since {report['tracking_since'][:10]}; earlier calls were not recorded.")
    else:
        notes.append("No model call has been recorded yet.")
    if report["agents_down"]:
        notes.append("Not running, so their numbers are missing: " + ", ".join(report["agents_down"]) + ".")
    sections.append({"text": " ".join(notes)})
    return {"title": "Usage across companies", "subtitle": "From records, no model was used for this answer.", "tone": "info", "sections": sections}


async def show(eng: Engine, conv: str, st: dict) -> None:
    tenant = st.get("tenant")
    if tenant is not None and not tenant.get("admin"):  # sign-in is on and this person is not an administrator
        return eng.say(conv, "Usage across companies is only shown to administrators. Ask yours if you need it.")
    report = await usage.build_report(eng)
    t = report["totals"]
    eng.say(conv, f"{_n(t['companies'])} compan{'y' if t['companies'] == 1 else 'ies'} and {_n(t['people'])} "
                  f"{'person' if t['people'] == 1 else 'people'} on this hub, {_n(t['calls'])} model calls so far "
                  f"({_n(t['input_tokens'] + t['output_tokens'])} tokens), {usage.human_bytes(t['storage_bytes'])} of data. Details by company:",
            cards=[card(report)])
