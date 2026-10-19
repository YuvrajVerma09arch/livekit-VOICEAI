"""Import-time configuration for the agent worker.

`load_dotenv()` runs exactly once here, so every other module reads an already
resolved environment. All tunables live in this one file.
"""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def env_float(name: str, default: float) -> float:
    """Read a float from the environment without crashing on a bad value."""
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Speech pipeline.
#
#   STT: sarvam realtime (language="auto" adaptive detection) primary
#   TTS: sarvam Bulbul v4 Flash (default) or self-hosted kokoro-82m
# ---------------------------------------------------------------------------
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_STT_LANGUAGE = "en"

# "auto" makes Sarvam adaptively identify the spoken language each turn
# (English / Hindi / Gujarati ...), so the transcript + LLM reply follow the
# user's language instead of being pinned to English. "transcribe" keeps the
# original language (translate/codemix would alter it).
SARVAM_STT_LANGUAGE = "auto"
SARVAM_STT_MODE = "transcribe"
# Sarvam's server-side end-of-turn silence (maps to silence_duration_ms) plus
# the framework's fixed commit floor (TURN_COMMIT_MIN_DELAY) form the turn-end
# gate: whichever is later wins, so both must move together. Raised to 450ms
# to match the new 0.45s commit floor; this gives non-native speakers extra
# breathing room. Revert to 300/0.3 if latency is unacceptable.
SARVAM_VAD_MIN_SILENCE_MS = 450
SARVAM_TTS_MODEL = "bulbul:v4-flash"
SARVAM_TTS_SPEAKER = "simran_enhi_customer"
SARVAM_TTS_LANGUAGE = "en-IN"

# TTS provider: "sarvam" (managed API, default), "kokoro" (self-hosted
# kokoro-82m HTTP service, see kokoro-tts/), or "elevenlabs" (managed API,
# supports Gujarati + Indian-accent voices via eleven_v3). Kokoro is ~82M
# params, 24kHz, English + Hindi (Devanagari auto-detected); it runs as a
# localhost sidecar now or on Render later.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sarvam")

# kokoro service settings, used only when TTS_PROVIDER == "kokoro"
KOKORO_TTS_URL = os.getenv("KOKORO_TTS_URL", "http://localhost:8000")
# which kokoro model the sidecar serves: "stock" (hexgrad 82M, af_heart /
# hf_alpha) or "goonj" (BH-Builds Hindi + Indian-English fine-tune with
# female persona voicepacks en_priya / hi_meera). Must match the sidecar's
# KOKORO_VARIANT - a mismatch is rejected loudly by the service instead of
# silently synthesizing with the wrong model.
KOKORO_TTS_VARIANT = os.getenv("KOKORO_TTS_VARIANT", "stock")
KOKORO_TTS_VOICE = os.getenv("KOKORO_TTS_VOICE") or (
    "en_priya" if KOKORO_TTS_VARIANT == "goonj" else "af_heart"
)
KOKORO_TTS_API_KEY = os.getenv("KOKORO_TTS_API_KEY", "")
KOKORO_TTS_SAMPLE_RATE = 24000  # kokoro's native output rate
KOKORO_TTS_TIMEOUT_S = 30.0

# ElevenLabs settings, used when TTS_PROVIDER == "elevenlabs" or
# STT_PROVIDER == "elevenlabs". Requires ELEVEN_API_KEY in .env.
# eleven_v3 is required for Gujarati (absent from multilingual_v2/flash_v2.5).
# Default voice_id is the plugin's safe built-in; swap for an Indian-accent
# voice from the ElevenLabs library to match the rest of the pipeline.
ELEVEN_TTS_VOICE_ID = os.getenv("ELEVEN_TTS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us")
ELEVEN_TTS_MODEL = os.getenv("ELEVEN_TTS_MODEL", "eleven_v3")
ELEVEN_STT_MODEL = os.getenv("ELEVEN_STT_MODEL", "scribe_v2_realtime")
ELEVEN_STT_SECONDARY_LANGUAGES = ["hi", "gu", "en"]

# STT provider: "sarvam" (realtime, language="auto" adaptive detection,
# default), "groq" (whisper primary + sarvam fallback), or "elevenlabs"
# (Scribe). Sarvam auto-detects Hindi/Gujarati/English so the reply language
# follows the user.
STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam")

# ---------------------------------------------------------------------------
# LLM fallback chain, in priority order: self-hosted Qwen (Kaggle notebook,
# OpenAI-compatible /v1 through a cloudflared tunnel) leads when configured,
# then Cloudflare Workers AI, then gemini flash family, then groq (qwen +
# gpt-oss) when everything above rate-limits or errors.
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]
GROQ_LLM_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
LLM_TEMPERATURE = 0.5

# Self-hosted Qwen lead: the Kaggle notebook (KAGGLE_QWEN_SELFHOST.ipynb)
# prints OLLAMA_BASE_URL (the tunnel's /v1) and OLLAMA_CLOUD_API_KEY at
# startup - paste both into .env. The trycloudflare URL changes every
# session, so update .env and restart the agent after each notebook restart.
# llama-server answers with whatever GGUF is loaded regardless of the model
# name, so OLLAMA_LLM_MODEL is cosmetic. Empty base URL or key skips the
# lead and Cloudflare stays at the front of the chain.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3-30b")
SELF_HOSTED_LLM_ENABLED = bool(OLLAMA_BASE_URL and OLLAMA_CLOUD_API_KEY)

# The fallback's attempt timeout gates every provider including this lead;
# 5s kills the T4 before its first token (TTFT through the tunnel is 3-15s)
# so the chain would silently fall through to Cloudflare every time. 15s
# clears it with margin, and a dead tunnel still fails fast on connect so
# fallback latency is unaffected when the notebook is off. Raise further via
# LLM_ATTEMPT_TIMEOUT_S if first tokens are still getting cut off.
LLM_ATTEMPT_TIMEOUT_S = env_float(
    "LLM_ATTEMPT_TIMEOUT_S", 15.0 if SELF_HOSTED_LLM_ENABLED else 5.0
)

# ---------------------------------------------------------------------------
# Cloudflare Workers AI (OpenAI-compatible /accounts/{id}/ai/v1 endpoint):
# keyed, leads the generation chain when no self-hosted Qwen is configured;
# missing account id or token skips it entirely, leaving gemini/groq as the
# chain. If the token lacks Workers AI permission it 401s and the chain
# falls through after the attempt timeout.
# ---------------------------------------------------------------------------
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_LLM_MODEL = os.getenv(
    "CLOUDFLARE_LLM_MODEL",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)
CLOUDFLARE_LLM_BASE_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    if CLOUDFLARE_ACCOUNT_ID
    else ""
)

# google rejects any manually-set deadline under 10s ("Minimum allowed deadline
# is 10s"); the fallback's attempt timeout is what otherwise feeds it. 15s
# clears the minimum with margin, so gemini no longer 400s on every request.
GEMINI_HTTP_TIMEOUT_MS = 15000

# cap what each LLM request carries to the last 10 messages (5 user/assistant
# exchanges); instructions are always kept. Bounds token spend and TTFT as a
# call gets longer - the provider never sees the full transcript.
HISTORY_MAX_MESSAGES = 10

# voice commit window: how long the session waits before treating silence as
# end-of-turn (Kilos never barges in, so this bounds every reply latency).
# Must stay in lockstep with SARVAM_VAD_MIN_SILENCE_MS - trimming one alone
# saves nothing since the later of the two gates the commit.
# 0.45s min gives hesitant or non-native speakers breathing room to gather
# their next word without triggering a premature commit; 3.5s max caps very
# long mid-sentence pauses. Loosened from 0.3/3.0 after reports of cut-offs.
TURN_COMMIT_MIN_DELAY = 0.45
TURN_COMMIT_MAX_DELAY = 3.5

# Silero VAD tuning (stock defaults are 0.05s min speech / 0.5 threshold, which
# let a click or an echo of Kilos's own voice open a turn - whisper then happily
# hallucinates "Thank you." out of 100ms of noise and the LLM answers it).
# 0.38 catches softer/quieter speakers and phone-quality microphones that the
# prior 0.45 missed, while still rejecting echo and background noise. 0.15s
# min speech keeps short genuine answers like "yes", "no", "haan" while
# rejecting sub-syllable blips and pops.
VAD_MIN_SPEECH_DURATION_S = 0.15
VAD_ACTIVATION_THRESHOLD = 0.38

# committed user turns whose transcript is only these (or has no letters/digits
# at all) are dropped before reaching the LLM instead of getting a real answer.
# Limited to true noise / filler syllables — legitimate words like "thank you"
# are kept so the agent can acknowledge them naturally.
IGNORED_USER_TURNS = frozenset(
    {
        "hmm",
        "hm",
        "huh",
        "uh",
        "um",
        "mmm",
        "ah",
        "oh",
    }
)

# cap how many tool round-trips one turn may burn before forcing an answer
MAX_TOOL_STEPS = 4

# fail a hung DuckDuckGo lookup fast instead of stalling the voice turn
WEB_SEARCH_TIMEOUT_S = 5.0

WEB_SEARCH_RESULTS = 5

LOG_NAME = "websearch-agent"
