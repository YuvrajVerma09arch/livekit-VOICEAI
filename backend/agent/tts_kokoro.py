"""Kokoro TTS adapter: talks to the self-hosted kokoro-tts HTTP service."""

import logging

import httpx
from livekit.agents import tts, utils
from livekit.agents.types import APIConnectOptions

from . import config

logger = logging.getLogger(config.LOG_NAME)

# HTTP status codes that are safe to retry (transient server-side issues)
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504, 429})
_MAX_RETRIES = 2


class KokoroTTS(tts.TTS):
    """Non-streaming TTS backed by the kokoro-tts FastAPI service.

    Declared non-streaming so the framework wraps it in StreamAdapter
    (sentence-level chunking); each sentence POSTs to the service and the
    returned WAV is emitted whole.
    """

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=config.KOKORO_TTS_SAMPLE_RATE,
            num_channels=1,
        )
        self._client = httpx.AsyncClient(
            base_url=config.KOKORO_TTS_URL,
            timeout=httpx.Timeout(config.KOKORO_TTS_TIMEOUT_S),
            headers=(
                {"Authorization": f"Bearer {config.KOKORO_TTS_API_KEY}"}
                if config.KOKORO_TTS_API_KEY
                else {}
            ),
        )

    @property
    def label(self) -> str:
        return f"kokoro-local({config.KOKORO_TTS_VARIANT}:{config.KOKORO_TTS_VOICE})"

    @property
    def model(self) -> str:
        return "kokoro-82m-goonj" if config.KOKORO_TTS_VARIANT == "goonj" else "kokoro-82m"

    @property
    def provider(self) -> str:
        return "kokoro-local"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return _KokoroChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        pass

    async def aclose(self) -> None:
        await self._client.aclose()


class _KokoroChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: KokoroTTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()

        last_exc: Exception | None = None
        resp: httpx.Response | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._tts._client.post(
                    "/v1/audio/speech",
                    json={
                        "text": self._input_text,
                        "voice": config.KOKORO_TTS_VOICE,
                        "variant": config.KOKORO_TTS_VARIANT,
                    },
                )
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    logger.warning(
                        f"kokoro TTS returned {resp.status_code} (attempt {attempt + 1}/"
                        f"{_MAX_RETRIES + 1}), retrying"
                    )
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"kokoro TTS timed out (attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying"
                    )
                    continue
                logger.error(f"kokoro TTS timed out after {_MAX_RETRIES + 1} attempts")
                raise
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.error(
                    f"kokoro TTS HTTP error {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                )
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"kokoro TTS request failed (attempt {attempt + 1}/"
                        f"{_MAX_RETRIES + 1}): {exc}"
                    )
                    continue
                logger.error(f"kokoro TTS failed after {_MAX_RETRIES + 1} attempts: {exc}")
                raise

        if resp is None:
            raise last_exc or RuntimeError("kokoro TTS produced no response")

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/wav",
        )
        output_emitter.push(resp.content)
        output_emitter.flush()
