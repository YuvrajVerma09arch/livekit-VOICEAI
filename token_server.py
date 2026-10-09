"""Small FastAPI service that hands short-lived LiveKit tokens to the web frontend.

Run it with:  uvicorn token_server:app --port 7860
"""

import ipaddress
import os
import re
import secrets
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel
from starlette.responses import Response

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
AVATAR_ENABLED = bool(os.getenv("SIMLI_API_KEY") and os.getenv("SIMLI_AVATAR_ID"))

TOKEN_TTL = timedelta(seconds=60 * 10)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# an unauthenticated token endpoint mints the only credential the agent accepts, so
# throttle it per caller - enough for a human retrying, useless for bulk abuse
TOKEN_RATE_LIMIT = _env_int("TOKEN_RATE_LIMIT", 10)
TOKEN_RATE_WINDOW = 60.0
# the endpoint only ever accepts a tiny JSON body; anything bigger is not a browser
MAX_TOKEN_BODY_BYTES = 4096

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(title="livekit-voice-agent token server")

_buckets: dict[str, deque[float]] = {}


def _client_ip(request: Request) -> str:
    """Caller IP for rate limiting.

    Behind Render's proxy the direct peer is an internal address, so in that case
    trust the rightmost X-Forwarded-For entry - the one appended by the proxy we
    are talking to - rather than the proxy's own address (which would put every
    visitor in the same bucket).
    """
    peer = request.client.host if request.client and request.client.host else ""
    if peer:
        try:
            if not ipaddress.ip_address(peer).is_private:
                return peer
        except ValueError:
            return peer

    forwarded = request.headers.get("x-forwarded-for", "")
    for entry in reversed([part.strip() for part in forwarded.split(",")]):
        if entry:
            return entry
    return peer or "unknown"


def _guard_response(request: Request) -> Response | None:
    """Refuse abusive /token requests before the body is read or the app runs."""
    if request.method != "POST" or request.url.path != "/token":
        return None

    # the endpoint only ever accepts a tiny JSON body
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_TOKEN_BODY_BYTES:
        return _error(413, "request body too large")

    # a browser request whose Origin does not match the host it reached us on is
    # cross-site abuse: the Vercel rewrite forwards the original Host, so the real
    # frontend passes. Non-browser clients send no Origin and stay rate limited.
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host", ""):
        return _error(403, "cross-origin requests not allowed")

    return None


def _error(status: int, detail: str) -> Response:
    return JSONResponse({"detail": detail}, status_code=status)


@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    guarded = _guard_response(request)
    response = guarded if guarded is not None else await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), geolocation=(), payment=()"
    )
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
    )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' blob: mediastream:; "
        "font-src 'self' data:; connect-src 'self' https: wss:; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'",
    )
    return response


def _throttle(request: Request) -> None:
    """Reject callers that request tokens faster than a human would."""
    ip = _client_ip(request)
    now = time.monotonic()
    bucket = _buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > TOKEN_RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= TOKEN_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="too many token requests, slow down")
    bucket.append(now)
    if len(_buckets) > 4096:
        for key in [k for k, v in _buckets.items() if not v]:
            del _buckets[key]


class TokenRequest(BaseModel):
    identity: str | None = None
    room: str | None = None


class TokenResponse(BaseModel):
    url: str
    token: str
    room: str
    identity: str
    avatar: bool


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "avatar": AVATAR_ENABLED}


@app.post("/token", response_model=TokenResponse)
def create_token(req: TokenRequest, request: Request) -> TokenResponse:
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set",
        )

    _throttle(request)

    room = req.room or f"room-{secrets.token_hex(4)}"
    identity = req.identity or f"user-{secrets.token_hex(4)}"

    # never echo a caller-controlled string back into a room or participant name
    if not _NAME_RE.match(room) or not _NAME_RE.match(identity):
        raise HTTPException(status_code=400, detail="invalid room or identity")

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(TOKEN_TTL)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    return TokenResponse(
        url=LIVEKIT_URL,
        token=token,
        room=room,
        identity=identity,
        avatar=AVATAR_ENABLED,
    )


# serve the production build of the frontend on the same origin when it exists,
# which keeps the token endpoint out of CORS territory
_dist = Path(__file__).parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
