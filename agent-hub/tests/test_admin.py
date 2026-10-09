"""The administrator's command line: companies, accounts, and that it never takes a password from the command line."""

from __future__ import annotations

import pytest

from agent_hub import admin
from agent_hub.auth import Auth
from agent_hub.store import Store

PASSWORD = "correct horse battery"


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    said: list[str] = []
    answers = iter([PASSWORD, PASSWORD] * 10)

    def run(*argv, ask=None):
        return admin.run(list(argv), ask=ask or (lambda _p: next(answers)), say=said.append, data_dir=tmp_path / "admin-data")

    run.said, run.dir = said, tmp_path / "admin-data"
    return run


def auth_for(cli) -> Auth:
    return Auth(Store(cli.dir))


def test_a_company_and_an_account_can_be_made_and_used(cli):
    assert cli("company", "add", "Accenture", "--deals", "acc-deals, acc-deals-2", "--rfp", "acc-rfp") == 0
    assert cli("user", "add", "dana@accenture.com", "--company", "Accenture", "--name", "Dana") == 0
    assert "Sign-in is now on" in cli.said[-1]
    _, user = auth_for(cli).login("dana@accenture.com", PASSWORD)
    assert user.allowed("deals") == ("acc-deals", "acc-deals-2") and user.allowed("rfp") == ("acc-rfp",)


def test_company_workspaces_can_be_changed_later(cli):
    cli("company", "add", "Accenture", "--deals", "a")
    cli("company", "set", "Accenture", "--rfp", "r1,r2")
    assert auth_for(cli).list_companies()[0]["workspaces"] == {"deals": ["a"], "rfp": ["r1", "r2"]}
    cli("company", "list")
    assert "Accenture: deals=a; rfp=r1,r2" in cli.said[-1]


def test_a_password_is_asked_twice_and_must_match(cli):
    cli("company", "add", "A")
    answers = iter(["first password!!", "different one!!!"])
    assert cli("user", "add", "a@b.com", "--company", "A", ask=lambda _p: next(answers)) == 1
    assert "not the same" in cli.said[-1] and auth_for(cli).list_users() == []


def test_a_short_password_and_a_duplicate_are_refused_with_a_message(cli):
    cli("company", "add", "A")
    short = iter(["short", "short"])
    assert cli("user", "add", "a@b.com", "--company", "A", ask=lambda _p: next(short)) == 1 and "at least 10" in cli.said[-1]
    assert cli("user", "add", "a@b.com", "--company", "A") == 0
    assert cli("user", "add", "a@b.com", "--company", "A") == 1 and "already has an account" in cli.said[-1]


def test_an_account_for_a_company_that_does_not_exist_is_refused(cli):
    assert cli("user", "add", "a@b.com", "--company", "Nobody") == 1 and "Create it first" in cli.said[-1]


def test_the_password_can_come_from_an_environment_variable_not_the_command_line(cli, monkeypatch):
    cli("company", "add", "A")
    monkeypatch.setenv("NEW_ACCOUNT_PASSWORD", PASSWORD)
    assert cli("user", "add", "a@b.com", "--company", "A", "--password-env", "NEW_ACCOUNT_PASSWORD") == 0
    with pytest.raises(SystemExit):  # there is deliberately no --password option
        cli("user", "add", "c@d.com", "--company", "A", "--password", PASSWORD)


def test_passwd_disable_enable_and_list(cli):
    cli("company", "add", "A")
    cli("user", "add", "a@b.com", "--company", "A")
    auth = auth_for(cli)
    token, _ = auth.login("a@b.com", PASSWORD)
    new = iter(["a brand new password", "a brand new password"])
    assert cli("user", "passwd", "a@b.com", ask=lambda _p: next(new)) == 0 and auth.user_for_token(token) is None
    cli("user", "disable", "a@b.com")
    cli("user", "list")
    assert "(disabled)" in cli.said[-1]
    cli("user", "enable", "a@b.com")
    assert auth_for(cli).login("a@b.com", "a brand new password")
    assert cli("user", "disable", "nobody@x.com") == 1


def test_listing_workspaces_works_when_the_agents_are_down(cli, monkeypatch):
    import httpx

    def refuse(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(admin.httpx, "get", refuse)
    assert cli("workspaces") == 0 and "not running" in " ".join(cli.said)


def test_an_administrator_can_be_made_from_the_command_line(cli):
    cli("company", "add", "A")
    assert cli("user", "add", "boss@b.com", "--company", "A", "--admin") == 0
    cli("user", "add", "dana@b.com", "--company", "A")
    cli("user", "list")
    assert "[admin]" in "\n".join(cli.said) and "dana@b.com" in cli.said[-1] and "[member]" in cli.said[-1]
    assert [u["role"] for u in auth_for(cli).list_users()] == ["admin", "member"]
