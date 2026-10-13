"""Request guards (early rejection) and security headers for the token service."""

from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

from . import config


def _error(status: int, detail: str) -> Response:
    return JSONResponse({"detail": detail}, status_code=status)


def _guard_response(request: Request) -> Response | None:
    """Refuse abusive /token requests before the body is read or the app runs."""
    if request.method != "POST" or request.url.path != "/token":
        return None

    # the endpoint only ever accepts a tiny JSON body
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > config.MAX_TOKEN_BODY_BYTES:
        return _error(413, "request body too large")

    # cross-site abuse guard: with TOKEN_ALLOWED_ORIGINS set, only listed browser
    # origins may call /token; otherwise fall back to matching Origin against the
    # Host it reached us on (the Vercel rewrite forwards the original Host, so the
    # real frontend passes). Non-browser clients send no Origin and stay rate limited.
    origin = request.headers.get("origin")
    if origin:
        netloc = urlsplit(origin).netloc
        if config.TOKEN_ALLOWED_ORIGINS:
            if netloc.lower() not in config.TOKEN_ALLOWED_ORIGINS:
                return _error(403, "origin not allowed")
        elif netloc != request.headers.get("host", ""):
            return _error(403, "cross-origin requests not allowed")

    return None


def install(app: FastAPI) -> None:
    """Register the guard + security-header middleware on `app`."""

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
