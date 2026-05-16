"""
Unit tests for auth helpers and protected routes.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from qsf.api.auth import (
    ALGORITHM,
    COOKIE_NAME,
    SECRET_KEY,
    create_session_token,
    decode_session_token,
)
from qsf.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def test_create_and_decode_session_token():
    token = create_session_token("user@example.com", "Test User")
    payload = decode_session_token(token)
    assert payload["sub"] == "user@example.com"
    assert payload["name"] == "Test User"


def test_decode_expired_token_raises():
    from jose import JWTError

    expired = {
        "sub": "user@example.com",
        "name": "Test",
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    token = jwt.encode(expired, SECRET_KEY, algorithm=ALGORITHM)
    with pytest.raises(JWTError):
        decode_session_token(token)


# ---------------------------------------------------------------------------
# /auth/me
# ---------------------------------------------------------------------------


def test_me_requires_auth():
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_returns_user_with_valid_cookie():
    token = create_session_token("user@example.com", "Test User")
    response = client.get("/auth/me", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"
    assert data["name"] == "Test User"


def test_me_rejects_invalid_token():
    response = client.get("/auth/me", cookies={COOKIE_NAME: "not.a.valid.token"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# /auth/logout
# ---------------------------------------------------------------------------


def test_logout_clears_cookie():
    token = create_session_token("user@example.com", "Test User")
    response = client.get(
        "/auth/logout",
        cookies={COOKIE_NAME: token},
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "/login.html" in response.headers["location"]
    # Cookie deletion appears as max-age=0 or expires in the past
    set_cookie = response.headers.get("set-cookie", "")
    assert COOKIE_NAME in set_cookie


# ---------------------------------------------------------------------------
# /auth/callback — CSRF state validation
# ---------------------------------------------------------------------------


def test_callback_rejects_mismatched_state():
    response = client.get(
        "/auth/callback?code=fake-code&state=wrong",
        cookies={"oauth_state": "correct"},
    )
    assert response.status_code == 400


def test_callback_rejects_missing_state_cookie():
    response = client.get("/auth/callback?code=fake-code&state=some-state")
    assert response.status_code == 400


def test_callback_rejects_missing_code():
    state = "test-state-abc"
    response = client.get(
        f"/auth/callback?state={state}",
        cookies={"oauth_state": state},
    )
    assert response.status_code == 400


def test_callback_sets_session_cookie(mocker):
    state = "test-state-xyz"

    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)
    mock_instance.fetch_token = AsyncMock(return_value={"access_token": "fake-token"})
    mock_userinfo = MagicMock()
    mock_userinfo.json.return_value = {"email": "user@example.com", "name": "Test User"}
    mock_instance.get = AsyncMock(return_value=mock_userinfo)

    mocker.patch("qsf.api.auth.AsyncOAuth2Client", return_value=mock_instance)

    response = client.get(
        f"/auth/callback?code=fake-code&state={state}",
        cookies={"oauth_state": state},
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert COOKIE_NAME in response.cookies


# ---------------------------------------------------------------------------
# /auth/login
# ---------------------------------------------------------------------------


def test_login_redirects_to_google_and_sets_state_cookie(mocker):
    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)
    mock_instance.create_authorization_url = MagicMock(
        return_value=("https://accounts.google.com/o/oauth2/v2/auth?client_id=test", "state123")
    )

    mocker.patch("qsf.api.auth.AsyncOAuth2Client", return_value=mock_instance)

    response = client.get("/auth/login", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert "accounts.google.com" in response.headers["location"]
    assert "oauth_state" in response.cookies
