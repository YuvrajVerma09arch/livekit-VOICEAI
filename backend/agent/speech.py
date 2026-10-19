"""STT / TTS builders for the configured speech pipeline."""

import logging

from livekit.agents import stt, tts, vad
from livekit.plugins import elevenlabs, groq, sarvam

from . import config

logger = logging.getLogger(config.LOG_NAME)


def build_stt(agent_vad: vad.VAD) -> stt.STT:
    if config.STT_PROVIDER == "sarvam":
        # language="auto" lets Sarvam adaptively identify the spoken language
        # each turn (English/Hindi/Gujarati), so the transcript and the LLM
        # reply follow the user's language instead of being pinned to English.
        # STTRealtime is the WebSocket endpoint (implicitly saaras realtime);
        # it takes mode, not model - the model kwarg only exists on batch STT.
        return sarvam.STTRealtime(
            language=config.SARVAM_STT_LANGUAGE,
            mode=config.SARVAM_STT_MODE,
            endpointing="vad",
            vad_min_silence_ms=config.SARVAM_VAD_MIN_SILENCE_MS,
        )

    if config.STT_PROVIDER == "elevenlabs":
        # Scribe is streaming and auto-detects language, so it handles
        # Hindi/Gujarati input without a pinned language or a VAD flush wrapper.
        return elevenlabs.STT(
            model=config.ELEVEN_STT_MODEL,
            include_language_detection=True,
            secondary_languages=config.ELEVEN_STT_SECONDARY_LANGUAGES,
        )

    whisper = groq.STT(
        model=config.GROQ_STT_MODEL,
        language=config.GROQ_STT_LANGUAGE,
    )
    sarvam_realtime = sarvam.STTRealtime(
        language=config.SARVAM_STT_LANGUAGE,
        endpointing="vad",
        vad_min_silence_ms=config.SARVAM_VAD_MIN_SILENCE_MS,
    )
    # whisper is non-streaming: the adapter needs the session VAD to flush turns
    return stt.FallbackAdapter(stt=[whisper, sarvam_realtime], vad=agent_vad)


def build_tts() -> tts.TTS:
    if config.TTS_PROVIDER == "elevenlabs":
        return elevenlabs.TTS(
            voice_id=config.ELEVEN_TTS_VOICE_ID,
            model=config.ELEVEN_TTS_MODEL,
        )

    if config.TTS_PROVIDER == "kokoro":
        from .tts_kokoro import KokoroTTS

        return KokoroTTS()

    return sarvam.TTS(
        target_language_code=config.SARVAM_TTS_LANGUAGE,
        model=config.SARVAM_TTS_MODEL,
        speaker=config.SARVAM_TTS_SPEAKER,
    )
