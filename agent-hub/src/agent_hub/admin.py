"""Make companies and accounts for the hub. Run from the agent-hub folder:

    .venv\\Scripts\\python.exe -m agent_hub.admin workspaces
    .venv\\Scripts\\python.exe -m agent_hub.admin company add "Accenture" --deals accenture-deals --rfp accenture-rfp
    .venv\\Scripts\\python.exe -m agent_hub.admin user add dana@accenture.com --company Accenture --name Dana
    .venv\\Scripts\\python.exe -m agent_hub.admin user list

A company is the workspaces it may use in each agent; a user belongs to one company and can reach nothing else. The
first account switches sign-in on (HUB_AUTH=auto). Passwords are asked for here, never taken from the command line, so
they do not end up in the shell history."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Callable
from pathlib import Path

import httpx

from .auth import Auth, AuthError
from .router import load_registry
from .store import Store


def _ids(raw: str | None) -> list[str] | None:
    return None if raw is None else [part.strip() for part in raw.split(",") if part.strip()]


def _workspaces_arg(args: argparse.Namespace) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for agent_id in ("deals", "rfp"):
        ids = _ids(getattr(args, agent_id))
        if ids is not None:
            found[agent_id] = ids
    return found


def _read_password(args: argparse.Namespace, ask: Callable[[str], str]) -> str:
    if getattr(args, "password_env", None):
        value = os.environ.get(args.password_env, "")
        if not value:
            raise AuthError("no_password", f"The environment variable {args.password_env} is empty.")
        return value
    first = ask("Password: ")
    if first != ask("Password again: "):
        raise AuthError("mismatch", "The two passwords are not the same.")
    return first


def _show_workspaces(say: Callable[[str], None]) -> None:
    for agent in load_registry():
        if agent.get("status") != "live":
            continue
        try:
            data = httpx.get(agent["url"] + "/v1/workspaces", timeout=5).json()
        except (httpx.HTTPError, ValueError):
            say(f"{agent['name']}: not running, so its workspaces can't be listed.")
            continue
        say(f"{agent['name']} (use these ids; --{'deals' if agent['id'] == 'deals' else 'rfp'}):")
        for space in data.get("workspaces", []):
            say(f"  {space['id']:<28} {space.get('name', '')} ({space.get('kind', '')})")


def run(argv: list[str], *, ask: Callable[[str], str] = getpass.getpass, say: Callable[[str], None] = print,
        data_dir: Path | str | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_hub.admin", description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="the hub's data folder (default: agent-hub/data)")
    top = parser.add_subparsers(dest="area", required=True)
    top.add_parser("workspaces", help="list each agent's workspace ids (read only)")

    company = top.add_parser("company").add_subparsers(dest="action", required=True)
    for name in ("add", "set"):
        sub = company.add_parser(name)
        sub.add_argument("name")
        sub.add_argument("--deals", help="comma-separated Deal Intelligence workspace ids")
        sub.add_argument("--rfp", help="comma-separated RFP Memory Assistant workspace ids")
    company.add_parser("list")

    user = top.add_parser("user").add_subparsers(dest="action", required=True)
    add = user.add_parser("add", allow_abbrev=False)  # so "--password" is an error, not "--password-env"
    add.add_argument("email")
    add.add_argument("--company", help="the customer company (not needed with --admin: administrators join Hub administrators)")
    add.add_argument("--name")
    add.add_argument("--admin", action="store_true", help="make this person an administrator (they can use the admin page)")
    add.add_argument("--password-env", help="read the password from this environment variable instead of asking")
    passwd = user.add_parser("passwd", allow_abbrev=False)
    passwd.add_argument("email")
    passwd.add_argument("--password-env")
    for name in ("disable", "enable"):
        user.add_parser(name).add_argument("email")
    user.add_parser("list")

    args = parser.parse_args(argv)
    if args.area == "workspaces":
        _show_workspaces(say)
        return 0
    store = Store(args.data_dir or data_dir)
    auth = Auth(store)
    try:
        if args.area == "company":
            if args.action == "add":
                auth.create_company(args.name, _workspaces_arg(args))
                say(f"Company {args.name!r} created.")
            elif args.action == "set":
                auth.set_company_workspaces(args.name, _workspaces_arg(args))
                say(f"Company {args.name!r} updated. Chats already open pick this up on their next message.")
            else:
                for row in auth.list_companies():
                    say(f"{row['name']}: " + "; ".join(f"{a}={','.join(ids) or '-'}" for a, ids in row["workspaces"].items()))
        else:
            if args.action == "add":
                if not args.admin and not args.company:
                    raise AuthError("bad_company", "A member needs --company (the customer company they work for).")
                auth.create_user(args.email, _read_password(args, ask), args.company or "", args.name, "admin" if args.admin else "member")
                say(f"Account {args.email} created for {args.company}. Sign-in is now {auth.mode()}.")
            elif args.action == "passwd":
                auth.set_password(args.email, _read_password(args, ask))
                say(f"Password changed for {args.email}; every session of theirs was ended.")
            elif args.action in ("disable", "enable"):
                auth.disable_user(args.email, args.action == "disable")
                say(f"{args.email} {args.action}d.")
            else:
                for row in auth.list_users():
                    say(f"{row['email']:<32} {row['company']}  [{row['role']}]{'  (disabled)' if row['disabled'] else ''}")
    except AuthError as exc:
        say(f"Error: {exc.message}")
        return 1
    finally:
        store.close()
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
