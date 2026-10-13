"""Environment configuration for the token service (loaded once at import)."""

import os
import re
from datetime import timedelta
from urllib.parse import urlsplit

from dotenv import load_dotenv

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

# never echo a caller-controlled string back into a room or participant name
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# shared secret for POST /token; while unset the endpoint fails closed (500)
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# origins allowed to call /token, scheme optional (e.g. https://my-app.vercel.app);
# empty = fall back to matching the request Host (local dev / same-origin hosting)
TOKEN_ALLOWED_ORIGINS = frozenset(
    host
    for entry in map(str.strip, os.getenv("TOKEN_ALLOWED_ORIGINS", "").split(","))
    if (host := urlsplit(entry if "://" in entry else f"//{entry}").netloc.lower())
)
