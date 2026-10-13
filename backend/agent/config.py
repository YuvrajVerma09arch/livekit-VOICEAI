"""Import-time configuration for the agent worker.

`load_dotenv()` runs exactly once here, so every other module reads an already
resolved environment. All tunables live in this one file.
"""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Speech pipeline.
#
#   STT: groq whisper-large-v3-turbo (primary) + sarvam realtime (auto-fallback)
#   TTS: sarvam Bulbul v4 Flash
# ---------------------------------------------------------------------------
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_STT_LANGUAGE = "en"

SARVAM_STT_LANGUAGE = "en-IN"
SARVAM_VAD_MIN_SILENCE_MS = 600
SARVAM_TTS_MODEL = "bulbul:v4-flash"
SARVAM_TTS_SPEAKER = "simran_enhi_customer"
SARVAM_TTS_LANGUAGE = "en-IN"

# ---------------------------------------------------------------------------
# LLM fallback chain, in priority order: gemini flash family first, then
# groq (qwen + gpt-oss) when gemini rate-limits.
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]
GROQ_LLM_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
LLM_TEMPERATURE = 0.5

# cap what each LLM request carries to the last 10 messages (5 user/assistant
# exchanges); instructions are always kept. Bounds token spend and TTFT as a
# call gets longer - the provider never sees the full transcript.
HISTORY_MAX_MESSAGES = 10

# voice commit window: how long the session waits before treating silence as
# end-of-turn (Kilos never barge in, so this bounds every reply latency)
TURN_COMMIT_MIN_DELAY = 0.4
TURN_COMMIT_MAX_DELAY = 3.0

# Silero VAD tuning (stock defaults are 0.05s min speech / 0.5 threshold, which
# let a click or an echo of Kilos's own voice open a turn - whisper then happily
# hallucinates "Thank you." out of 100ms of noise and the LLM answers it).
# Require a real voiced utterance before STT ever sees the audio; 0.2s keeps
# short genuine answers like "yes" while rejecting blips.
VAD_MIN_SPEECH_DURATION_S = 0.2
VAD_ACTIVATION_THRESHOLD = 0.55

# committed user turns whose transcript is only these (or has no letters/digits
# at all) are dropped before reaching the LLM instead of getting a real answer
IGNORED_USER_TURNS = frozenset(
    {
        "thank you",
        "thankyou",
        "thanks",
        "thank u",
        "thx",
        "thank you so much",
        "thanks a lot",
        "hmm",
        "hm",
        "huh",
        "uh",
        "um",
        "mmm",
    }
)

# cap how many tool round-trips one turn may burn before forcing an answer
MAX_TOOL_STEPS = 4

# fail a hung DuckDuckGo lookup fast instead of stalling the voice turn
WEB_SEARCH_TIMEOUT_S = 5.0

WEB_SEARCH_RESULTS = 5

LOG_NAME = "websearch-agent"


def env_float(name: str, default: float) -> float:
    """Read a float from the environment without crashing on a bad value."""
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default
