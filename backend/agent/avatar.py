"""Avatar session — provider-agnostic.

An avatar provider joins the room as a second participant and publishes the
synchronized audio + video tracks. When one is active, the agent's own audio
output is routed to it instead of the room (see the audio_output flag on
session.start in server.py). Providers share the same three-call surface —
start / wait_for_join / aclose — so the dispatcher below is identical for both.

Providers:
    simli  - 3D cartoon face (SIMLI_API_KEY + SIMLI_AVATAR_ID)
    tavus  - photoreal human face (TAVUS_API_KEY; not 2D)
    none   - voice-only (no avatar participant)

Selection is driven by AVATAR_PROVIDER ("simli" | "tavus" | "none"); when it is
unset we auto-detect from whichever credentials are present (simli wins if both
are set). Any start failure logs and falls back to voice-only rather than
leaving the user with dead air.
"""

import logging
import os
from typing import Literal, Protocol

from livekit.agents import AgentSession, JobContext
from livekit.plugins import simli, tavus

from . import config

logger = logging.getLogger(config.LOG_NAME)

Provider = Literal["simli", "tavus", "none"]


class _AvatarSession(Protocol):
    async def start(self, session: AgentSession, room) -> None: ...
    async def wait_for_join(self, *, timeout: float | None = None) -> None: ...
    async def aclose(self) -> None: ...


def _simli_enabled() -> bool:
    return bool(os.getenv("SIMLI_API_KEY") and os.getenv("SIMLI_AVATAR_ID"))


def _tavus_enabled() -> bool:
    return bool(os.getenv("TAVUS_API_KEY"))


def resolve_provider() -> Provider:
    """Pick the avatar provider from AVATAR_PROVIDER, or auto-detect."""
    explicit = os.getenv("AVATAR_PROVIDER", "").strip().lower()
    if explicit in ("simli", "tavus", "none"):
        return explicit
    if _simli_enabled():
        return "simli"
    if _tavus_enabled():
        return "tavus"
    return "none"


def avatar_enabled() -> bool:
    return resolve_provider() != "none"


def _build_simli() -> _AvatarSession:
    return simli.AvatarSession(
        simli_config=simli.SimliConfig(
            api_key=os.environ["SIMLI_API_KEY"],
            face_id=os.environ["SIMLI_AVATAR_ID"],
            # keep the avatar alive while the user is thinking instead of dropping
            # it after Simli's 30s idle default
            max_session_length=int(os.getenv("SIMLI_MAX_SESSION_LENGTH", "1800")),
            max_idle_time=int(os.getenv("SIMLI_MAX_IDLE_TIME", "300")),
        )
    )


def _build_tavus() -> _AvatarSession:
    kwargs: dict[str, str] = {}
    # both optional: unset face_id/pal_id falls back to the stock "Lucy - Home" face
    if os.getenv("TAVUS_FACE_ID"):
        kwargs["face_id"] = os.environ["TAVUS_FACE_ID"]
    if os.getenv("TAVUS_PAL_ID"):
        kwargs["pal_id"] = os.environ["TAVUS_PAL_ID"]
    return tavus.AvatarSession(**kwargs)


async def start_avatar(
    session: AgentSession, ctx: JobContext
) -> _AvatarSession | None:
    """Start the avatar; returns None (voice-only) on any failure."""
    provider = resolve_provider()
    if provider == "none":
        logger.warning("no avatar credentials set, starting voice-only")
        return None

    if provider == "simli":
        join_timeout = config.env_float("SIMLI_JOIN_TIMEOUT", 15.0)
        candidate = _build_simli()
    else:
        join_timeout = config.env_float("TAVUS_JOIN_TIMEOUT", 30.0)
        candidate = _build_tavus()

    # the avatar joins the room before the agent session starts, so connect first
    # (session.start() would otherwise connect lazily and wait_for_join would no-op)
    await ctx.connect()
    try:
        await candidate.start(session, room=ctx.room)
        # wait for the avatar video track so the user never hears audio with no
        # video. Both providers join in a few seconds normally, but a failed
        # connection (bad id, quota, outage) can hang — this timeout is what bounds
        # it before we fall back to voice-only instead of leaving dead air.
        await candidate.wait_for_join(timeout=join_timeout)
        logger.info(f"{provider} avatar joined the room")
        return candidate
    except TimeoutError:
        logger.warning(
            f"{provider} avatar did not join within {join_timeout}s, "
            "falling back to voice-only output"
        )
        await candidate.aclose()
    except Exception:
        logger.exception(
            f"failed to start {provider} avatar, falling back to voice-only output"
        )
        await candidate.aclose()
    return None
