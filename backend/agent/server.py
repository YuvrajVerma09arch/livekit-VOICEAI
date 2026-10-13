"""AgentServer wiring: session construction and the LiveKit entrypoint.

Run with `python -m backend.agent start`.
"""

import logging

from livekit.agents import (
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    SessionUsageUpdatedEvent,
    cli,
    inference,
    room_io,
)

from . import config
from .agent import KilosAgent
from .avatar import start_avatar
from .llm import build_llm
from .speech import build_stt, build_tts

logger = logging.getLogger(config.LOG_NAME)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    agent_vad = inference.VAD(
        model="silero",
        min_speech_duration=config.VAD_MIN_SPEECH_DURATION_S,
        activation_threshold=config.VAD_ACTIVATION_THRESHOLD,
    )
    speech_stt = build_stt(agent_vad)
    speech_tts = build_tts()
    logger.info(
        f"speech pipeline: stt={speech_stt.label} tts={speech_tts.label}"
    )

    session: AgentSession = AgentSession(
        # no barge-in: Kilos always finishes what it is saying. Audio heard while it
        # talks is dropped rather than queued (the default discard_audio_if_uninterruptible)
        turn_handling={
            "interruption": {"enabled": False},
            "endpointing": {
                "mode": "fixed",
                "min_delay": config.TURN_COMMIT_MIN_DELAY,
                "max_delay": config.TURN_COMMIT_MAX_DELAY,
            },
            "preemptive_generation": {
                "enabled": False,
            },
        },
        vad=agent_vad,
        max_tool_steps=config.MAX_TOOL_STEPS,
        stt=speech_stt,
        llm=build_llm(),
        tts=speech_tts,
        # word-synced captions, only offered when the TTS reports alignment
        use_tts_aligned_transcript=speech_tts.capabilities.aligned_transcript,
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
        ],
    )

    @session.on("session_usage_updated")
    def _on_session_usage(ev: SessionUsageUpdatedEvent) -> None:
        logger.info(f"session usage: {ev.usage}")

    @session.on("conversation_item_added")
    def _on_conversation_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        turn_metrics = getattr(item, "metrics", None)
        if getattr(item, "role", None) != "assistant" or turn_metrics is None:
            return
        logger.info(f"turn metrics: {turn_metrics}")

    async def log_usage() -> None:
        logger.info(f"Usage: {session.usage}")

    ctx.add_shutdown_callback(log_usage)

    avatar = await start_avatar(session, ctx)

    await session.start(
        agent=KilosAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            # when the avatar is active it publishes the agent audio, so the agent must
            # not publish its own audio track (otherwise both would be heard)
            audio_output=avatar is None,
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
