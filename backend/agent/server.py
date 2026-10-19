"""AgentServer wiring: session construction and the LiveKit entrypoint.

Run with `python -m backend.agent start`.
"""

import logging

from livekit.agents import (
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    SessionUsageUpdatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
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
        turn_handling={
            # "vad" disables the framework's auto-selected EOT turn detector: that
            # model awaits a cloud end-of-turn prediction (up to 1s timeout) before
            # committing every turn, and can stretch the commit to max_delay (3s)
            # when it doubts the turn ended. Pure VAD + the fixed delay below is
            # both faster per-turn and deterministic.
            "turn_detection": "vad",
            # barge-in via LiveKit's adaptive (ML) overlapping-speech classifier. This
            # also fixes silent-after-first-reply: with interruptions off, the framework
            # discards ALL mic audio while it believes the agent is speaking (the
            # discard_audio_if_uninterruptible default), so any user speech during or
            # after a long reply vanished with no error. min_duration 0.5s +
            # false_interruption resume (2s) keep coughs/backchannels from killing replies.
            "interruption": {"enabled": True, "mode": "adaptive"},
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

    # no-reply-bug diagnostics: pin each stage of the input -> transcript ->
    # commit -> reply pipeline to the log so a silent session shows where
    # the chain stops instead of showing nothing.
    @session.on("user_input_transcribed")
    def _on_user_input(ev: UserInputTranscribedEvent) -> None:
        logger.info(
            f"user input: final={ev.is_final} lang={ev.language} "
            f"speaker={ev.speaker_id} text={ev.transcript!r}"
        )

    @session.on("user_transcription_timeout")
    def _on_user_transcription_timeout(ev: UserTranscriptionTimeoutEvent) -> None:
        logger.warning(
            f"user transcription timeout: speech_duration={ev.speech_duration}s "
            f"vad_speech_started_at={ev.vad_speech_started_at}"
        )

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        logger.info(f"user state: {ev.old_state} -> {ev.new_state}")

    @session.on("agent_state_changed")
    def _on_agent_state(ev: AgentStateChangedEvent) -> None:
        logger.info(f"agent state: {ev.old_state} -> {ev.new_state}")

    @session.on("error")
    def _on_error(ev: ErrorEvent) -> None:
        logger.error(f"session error: {ev.error}")

    async def log_usage() -> None:
        logger.info(f"Usage: {session.usage}")

    ctx.add_shutdown_callback(log_usage)

    avatar = await start_avatar(session, ctx)

    await session.start(
        agent=KilosAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # when the avatar is active it publishes the agent audio, so the agent must
            # not publish its own audio track (otherwise both would be heard)
            audio_output=avatar is None,
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
