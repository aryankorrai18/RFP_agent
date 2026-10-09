"""Accounts, companies, sessions and sign-in limits, against a real (temporary) database."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from agent_hub import auth as auth_module
from agent_hub.auth import Auth, AuthError, hash_password, verify_password
from agent_hub.store import Store

PASSWORD = "correct horse battery"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "auth-data")
    yield s
    s.close()


@pytest.fixture
def auth(store, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    a = Auth(store)
    a.create_company("Accenture", {"deals": ["acc-deals"], "rfp": ["acc-rfp", "acc-rfp-2"]})
    a.create_company("Globex", {"deals": ["globex-deals"]})
    a.create_user("Dana@Accenture.com", PASSWORD, "Accenture", "Dana")
    return a


def test_passwords_are_salted_scrypt_hashes():
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and first.startswith("scrypt$") and PASSWORD not in first
    assert verify_password(PASSWORD, first) and not verify_password("wrong password!!", first)
    assert not verify_password(PASSWORD, "garbage") and not verify_password(PASSWORD, "scrypt$x$y")


def test_signing_in_returns_the_user_and_a_session_that_works(auth):
    token, user = auth.login("dana@accenture.com", PASSWORD, "1.2.3.4")
    assert (user.email, user.company_name, user.name) == ("dana@accenture.com", "Accenture", "Dana")
    assert user.allowed("deals") == ("acc-deals",) and user.allowed("rfp") == ("acc-rfp", "acc-rfp-2")
    assert auth.user_for_token(token).id == user.id
    assert auth.user_for_token("not-a-token") is None and auth.user_for_token(None) is None


def test_email_case_does_not_matter(auth):
    assert auth.login("DANA@ACCENTURE.COM", PASSWORD)


def test_unknown_and_wrong_give_the_same_message_and_status(auth):
    errors = []
    for email, password in (("dana@accenture.com", "wrong password!!"), ("nobody@x.com", "wrong password!!")):
        with pytest.raises(AuthError) as exc:
            auth.login(email, password)
        errors.append((exc.value.code, exc.value.status, exc.value.message))
    assert errors[0] == errors[1] == ("bad_login", 401, "That email and password do not match an account.")


def test_only_the_hash_of_a_session_token_is_stored(auth, store):
    token, _ = auth.login("dana@accenture.com", PASSWORD)
    stored = [row["token_hash"] for row in store.sql_all("SELECT token_hash FROM sessions")]
    assert stored and token not in stored and all(len(h) == 64 for h in stored)


def test_logout_ends_the_session(auth):
    token, _ = auth.login("dana@accenture.com", PASSWORD)
    auth.logout(token)
    assert auth.user_for_token(token) is None


def test_an_expired_session_is_refused_and_removed(auth, store):
    token, _ = auth.login("dana@accenture.com", PASSWORD)
    store.sql_exec("UPDATE sessions SET expires_at = ?", ("2000-01-01T00:00:00+00:00",))
    assert auth.user_for_token(token) is None and store.sql_all("SELECT 1 FROM sessions") == []


def test_five_failures_lock_that_email_and_address_out_for_a_while(auth):
    for _ in range(5):
        with pytest.raises(AuthError):
            auth.login("dana@accenture.com", "wrong password!!", "9.9.9.9")
    with pytest.raises(AuthError) as locked:
        auth.login("dana@accenture.com", PASSWORD, "9.9.9.9")  # even the right password is refused now
    assert locked.value.status == 429 and locked.value.retry_after
    assert auth.login("dana@accenture.com", PASSWORD, "5.5.5.5")  # another address is not locked


def test_a_good_login_clears_the_failures(auth):
    for _ in range(4):
        with pytest.raises(AuthError):
            auth.login("dana@accenture.com", "wrong password!!", "1.1.1.1")
    auth.login("dana@accenture.com", PASSWORD, "1.1.1.1")
    for _ in range(4):
        with pytest.raises(AuthError):
            auth.login("dana@accenture.com", "wrong password!!", "1.1.1.1")


def test_old_failures_stop_counting(auth, store):
    for _ in range(5):
        with pytest.raises(AuthError):
            auth.login("dana@accenture.com", "wrong password!!", "2.2.2.2")
    store.sql_exec("UPDATE login_failures SET at = ?", ((auth_module._now() - timedelta(hours=1)).isoformat(timespec="seconds"),))
    assert auth.login("dana@accenture.com", PASSWORD, "2.2.2.2")


def test_a_disabled_user_cannot_sign_in_and_loses_the_session(auth):
    token, _ = auth.login("dana@accenture.com", PASSWORD)
    auth.disable_user("dana@accenture.com")
    assert auth.user_for_token(token) is None
    with pytest.raises(AuthError):
        auth.login("dana@accenture.com", PASSWORD)
    auth.disable_user("dana@accenture.com", False)
    assert auth.login("dana@accenture.com", PASSWORD)


def test_a_new_password_ends_every_session(auth):
    token, _ = auth.login("dana@accenture.com", PASSWORD)
    auth.set_password("dana@accenture.com", "a brand new password")
    assert auth.user_for_token(token) is None
    with pytest.raises(AuthError):
        auth.login("dana@accenture.com", PASSWORD)
    assert auth.login("dana@accenture.com", "a brand new password")


@pytest.mark.parametrize("email, password, code", [
    ("not-an-email", PASSWORD, "bad_email"), ("x@y.com", "short", "weak_password"), ("dana@accenture.com", PASSWORD, "user_exists")])
def test_account_rules(auth, email, password, code):
    with pytest.raises(AuthError) as exc:
        auth.create_user(email, password, "Accenture")
    assert exc.value.code == code


def test_company_rules(auth):
    with pytest.raises(AuthError) as exc:
        auth.create_company("accenture")
    assert exc.value.code == "company_exists"
    with pytest.raises(AuthError) as bad:
        auth.create_company("Initech", {"payroll": ["x"]})
    assert bad.value.code == "bad_company"
    with pytest.raises(AuthError) as none:
        auth.create_user("a@b.com", PASSWORD, "No such company")
    assert none.value.code == "no_company"


def test_changing_a_companys_workspaces_applies_to_the_next_sign_in(auth):
    auth.set_company_workspaces("Accenture", {"deals": ["acc-deals", "acc-deals-2"]})
    _, user = auth.login("dana@accenture.com", PASSWORD)
    assert user.allowed("deals") == ("acc-deals", "acc-deals-2") and user.allowed("rfp") == ("acc-rfp", "acc-rfp-2")


def test_a_user_gets_their_own_companys_workspaces_only(auth):
    auth.create_user("gus@globex.com", PASSWORD, "Globex")
    _, gus = auth.login("gus@globex.com", PASSWORD)
    assert gus.allowed("deals") == ("globex-deals",) and gus.allowed("rfp") == ()


@pytest.mark.parametrize("setting, expected", [("on", "on"), ("off", "off"), ("auto", "on"), (None, "on")])
def test_the_mode_follows_hub_auth(auth, monkeypatch, setting, expected):
    if setting:
        monkeypatch.setenv("HUB_AUTH", setting)
    assert auth.mode() == expected


def test_auto_mode_is_open_until_the_first_account_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    fresh = Auth(Store(tmp_path / "fresh"))
    assert fresh.mode() == "off"
    fresh.create_company("A")
    assert fresh.mode() == "off"
    fresh.create_user("a@b.com", PASSWORD, "A")
    assert fresh.mode() == "on"
    fresh.disable_user("a@b.com")
    assert fresh.mode() == "off"
    fresh.store.close()


def test_a_chat_remembers_its_owner_and_old_chats_have_none(store):
    mine = store.create_conversation("user-1")
    assert store.conversation_owner(mine) == "user-1" and store.conversation_owner(store.create_conversation()) is None
    assert store.conversation_owner("missing") is None


def test_a_database_from_before_sign_in_is_upgraded(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    old = sqlite3.connect(folder / "hub.db")
    old.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, state TEXT NOT NULL)")
    old.execute("INSERT INTO conversations VALUES ('c1', 'now', '{}')")
    old.commit()
    old.close()
    store = Store(folder)
    assert store.conversation_owner("c1") is None and store.conversation_exists("c1")
    assert store.conversation_owner(store.create_conversation("u")) == "u"
    store.close()
