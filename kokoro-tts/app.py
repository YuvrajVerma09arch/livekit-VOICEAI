"""Kokoro TTS service: POST /v1/audio/speech -> 24kHz mono WAV.

Serves ONE kokoro-82M variant per process, selected by KOKORO_VARIANT:

  stock (default): hexgrad/Kokoro-82M - English via af_heart, Hindi via
      hf_alpha. Original behavior, unchanged.
  goonj: BH-Builds/goonj-1-82M - Hindi + Indian-English fine-tune with
      female-first persona voicepacks (en_priya, hi_meera, ...); same
      architecture/vocab as stock, so only the weights differ.

Devanagari text routes to the Hindi pipeline, everything else to the
English pipeline. Switching variants = change KOKORO_VARIANT and restart;
only one model is resident at a time (~330MB).
"""

import io
import logging
import os
import re
import threading
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from kokoro import KPipeline
from kokoro.model import KModel
from pydantic import BaseModel

logger = logging.getLogger("kokoro-tts")

SAMPLE_RATE = 24000
STOCK_REPO = "hexgrad/Kokoro-82M"
GOONJ_REPO = "BH-Builds/goonj-1-82M"
GOONJ_WEIGHTS = "kokoro_hindi_final.pth"

# optional bearer auth for public hosting; unset = no auth (local sidecar).
# the agent already sends this via KOKORO_TTS_API_KEY when configured.
API_KEY = os.getenv("KOKORO_API_KEY", "")

VARIANT = os.getenv("KOKORO_VARIANT", "stock")
if VARIANT not in ("stock", "goonj"):
    raise RuntimeError(f"KOKORO_VARIANT must be 'stock' or 'goonj', got {VARIANT!r}")

# Devanagari block (U+0900-U+097F); Hindi text is written in this script, so
# its presence is a reliable signal to route to the Hindi pipeline.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

if VARIANT == "goonj":
    REPO_ID = GOONJ_REPO
    # goonj personas are named en_*/hi_*; defaults pick the female voices so
    # the assistant matches its female persona without extra config
    _DEFAULT_VOICES = {"en": "en_priya", "hi": "hi_meera"}
else:
    REPO_ID = STOCK_REPO
    _DEFAULT_VOICES = {"en": "af_heart", "hi": "hf_alpha"}

# a client voice must belong to the routed language's family so an English
# voice never speaks Hindi (and vice versa); anything else falls back to the
# per-language default. goonj's bed_hindi is the Hindi language-bed voice.
if VARIANT == "goonj":
    _VOICE_PREFIXES = {"en": ("en_",), "hi": ("hi_", "bed_")}
else:
    # stock Kokoro voice names are <accent><gender>_<name>
    _VOICE_PREFIXES = {"en": ("af_", "am_", "bf_", "bm_"), "hi": ("hf_", "hm_")}

# one shared KModel so the two language pipelines don't double the ~330MB
# footprint; each KPipeline only differs by its G2P (espeak 'hi' vs English)
_model: KModel | None = None
_pipelines: dict[str, KPipeline] = {}

# MPS is not thread-safe: FastAPI runs this sync endpoint in a threadpool, so
# concurrent requests (sentence-chunked replies, overlapping turns) race the
# same Metal command buffer and abort the whole process with an AGX
# command-encoder assertion. One synth at a time.
_synth_lock = threading.Lock()


def _load_model() -> None:
    global _model

    if VARIANT == "goonj":
        model_path = hf_hub_download(GOONJ_REPO, GOONJ_WEIGHTS)
        config_path = hf_hub_download(GOONJ_REPO, "config.json")
    else:
        model_path = hf_hub_download(STOCK_REPO, "kokoro-v1_0.pth")
        config_path = hf_hub_download(STOCK_REPO, "config.json")

    # Apple Silicon (MPS) synthesizes kokoro far faster than CPU, but some
    # StyleTTS2 ops can be missing on MPS. Smoke-test a tiny synth first and
    # fall back to CPU instead of crashing at the first real request.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = KModel(repo_id=REPO_ID, config=config_path, model=model_path).to(device).eval()
    if device == "mps":
        try:
            pipe = KPipeline(lang_code="a", model=model, repo_id=REPO_ID)
            next(pipe("hi", voice=_DEFAULT_VOICES["en"]), None)
        except Exception:
            logger.warning("MPS smoke test failed; falling back to CPU", exc_info=True)
            device = "cpu"
            model = KModel(repo_id=REPO_ID, config=config_path, model=model_path).to(device).eval()

    _model = model
    _pipelines["en"] = KPipeline(lang_code="a", model=_model, repo_id=REPO_ID)
    _pipelines["hi"] = KPipeline(lang_code="h", model=_model, repo_id=REPO_ID)
    # fetch the default voicepacks now so the first live request
    # doesn't pay the download
    _pipelines["en"].load_voice(_DEFAULT_VOICES["en"])
    _pipelines["hi"].load_voice(_DEFAULT_VOICES["hi"])
    logger.info(f"kokoro-82m ({VARIANT}) loaded on {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="kokoro-tts", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class SpeechRequest(BaseModel):
    text: str
    voice: str | None = None
    lang: str | None = None
    variant: str | None = None


def _detect_lang(text: str) -> str:
    return "hi" if _DEVANAGARI.search(text) else "en"


def _split_hindi(text: str) -> str:
    # break on Hindi danda (।) and sentence-ending punctuation so each espeak
    # chunk stays short enough not to be truncated
    return re.sub(r"(?<=[।!?])\s+", "\n", text)


@app.post("/v1/audio/speech")
def speech(
    req: SpeechRequest, authorization: str | None = Header(default=None)
) -> Response:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    if not _pipelines:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    # agent and sidecar must agree on which model is loaded; a silent
    # mismatch would synthesize with the wrong weights under the right voice
    if req.variant is not None and req.variant != VARIANT:
        raise HTTPException(
            status_code=422,
            detail=f"variant mismatch: service serves {VARIANT!r}, request asked {req.variant!r}",
        )

    lang = req.lang if req.lang in _pipelines else _detect_lang(req.text)
    pipeline = _pipelines[lang]

    # only honour an explicit voice if it belongs to this language's family
    # (hf_*/hm_* vs af_*/am_*); otherwise fall back to the per-language default
    prefix_ok = req.voice is not None and req.voice.startswith(_VOICE_PREFIXES[lang])
    voice = req.voice if prefix_ok else _DEFAULT_VOICES[lang]

    # Hindi (espeak G2P) truncates long text unless split on newlines; English
    # pipelines chunk fine on their own, so only Hindi needs the split.
    text = _split_hindi(req.text) if lang == "hi" else req.text

    # collect every chunk as a numpy array first, then write ONE wav at the
    # end. chunk.audio is a torch Tensor (not numpy), and appending multiple
    # separate WAV files into one BytesIO corrupts the buffer - that was the
    # LibsndfileError on multi-chunk (Hindi) requests.
    pieces: list[np.ndarray] = []
    with _synth_lock:
        for chunk in pipeline(text, voice=voice):
            audio = getattr(chunk, "audio", None)
            if audio is None or len(audio) == 0:
                continue
            pieces.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))

    if not pieces:
        raise HTTPException(status_code=500, detail="no audio produced")

    buf = io.BytesIO()
    sf.write(buf, np.concatenate(pieces), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
