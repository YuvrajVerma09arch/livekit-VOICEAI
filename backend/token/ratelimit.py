"""Per-caller rate limiting for POST /token."""

import ipaddress
import time
from collections import deque

from fastapi import HTTPException, Request

from . import config

# ip -> timestamps of recent accepted requests
_buckets: dict[str, deque[float]] = {}


def client_ip(request: Request) -> str:
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


def throttle(request: Request) -> None:
    """Reject callers that request tokens faster than a human would."""
    ip = client_ip(request)
    now = time.monotonic()
    bucket = _buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > config.TOKEN_RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= config.TOKEN_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="too many token requests, slow down")
    bucket.append(now)
    if len(_buckets) > 4096:
        for key in [k for k, v in _buckets.items() if not v]:
            del _buckets[key]
