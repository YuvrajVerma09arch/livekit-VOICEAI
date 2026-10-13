"""STT / TTS builders for the configured speech pipeline."""

import logging

from livekit.agents import stt, tts, vad
from livekit.plugins import groq, sarvam

from . import config

logger = logging.getLogger(config.LOG_NAME)


def build_stt(agent_vad: vad.VAD) -> stt.STT:
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
    return sarvam.TTS(
        target_language_code=config.SARVAM_TTS_LANGUAGE,
        model=config.SARVAM_TTS_MODEL,
        speaker=config.SARVAM_TTS_SPEAKER,
    )
