from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets


TOKEN_TTL_HOURS = 8


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Generate a PBKDF2 value for `COCOA_ADMIN_PASSWORD_HASH` setup."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return "pbkdf2_sha256$310000${}${}".format(
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iteration_text, salt_text, expected_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(expected_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iteration_text))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def issue_token(email: str, secret: str) -> str:
    payload = {
        "sub": email,
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)).timestamp()),
    }
    encoded_payload = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    signature = hmac.new(secret.encode(), encoded_payload.encode(), hashlib.sha256).hexdigest()
    return f"{encoded_payload}.{signature}"


def validate_token(token: str, secret: str) -> str | None:
    try:
        encoded_payload, signature = token.rsplit(".", 1)
        expected = hmac.new(secret.encode(), encoded_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload.encode()))
        if int(payload["exp"]) <= int(datetime.now(timezone.utc).timestamp()):
            return None
        return str(payload["sub"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
