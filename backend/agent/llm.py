"""LLM builder - the single place the language model is configured."""

import logging
from typing import Any

from google.genai import types as genai_types
from livekit.agents import llm
from livekit.agents.llm.fallback_adapter import DEFAULT_FALLBACK_API_CONNECT_OPTIONS
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.plugins import google, groq, openai

from . import config

logger = logging.getLogger(config.LOG_NAME)


def _trim_chat_ctx(chat_ctx: llm.ChatContext, max_messages: int) -> llm.ChatContext:
    """Keep instructions plus the last ``max_messages`` user/assistant messages.

    A function-call output whose call was trimmed away is dropped too: providers
    reject outputs that reference a call they never saw.
    """
    items = chat_ctx.items
    conversational = [
        i
        for i, it in enumerate(items)
        if it.type == "message" and it.role in ("user", "assistant")
    ]
    if len(conversational) <= max_messages:
        return chat_ctx

    boundary = conversational[-max_messages]
    kept = [
        it
        for i, it in enumerate(items)
        if i >= boundary
        or (it.type == "message" and it.role not in ("user", "assistant"))
    ]
    kept_calls = {it.call_id for it in kept if it.type == "function_call"}
    kept = [
        it
        for it in kept
        if it.type != "function_call_output" or it.call_id in kept_calls
    ]
    return llm.ChatContext(kept)


class HistoryCappedFallback(llm.FallbackAdapter):
    """FallbackAdapter that trims history so each request carries only the last
    ``HISTORY_MAX_MESSAGES`` exchanges instead of the full transcript."""

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return super().chat(
            chat_ctx=_trim_chat_ctx(chat_ctx, config.HISTORY_MAX_MESSAGES),
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )


def build_llm() -> llm.LLM:
    chain: list[llm.LLM] = []
    # Self-hosted Qwen (Kaggle llama-server via cloudflared tunnel) leads when
    # configured; skipped entirely when OLLAMA_BASE_URL / OLLAMA_CLOUD_API_KEY
    # are absent. llama-server accepts any model name, so the label is cosmetic.
    if config.SELF_HOSTED_LLM_ENABLED:
        chain.append(
            openai.LLM(
                model=config.OLLAMA_LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                api_key=config.OLLAMA_CLOUD_API_KEY,
                temperature=config.LLM_TEMPERATURE,
            )
        )
    # Cloudflare Workers AI is the next lead when no self-hosted Qwen is
    # configured; skipped entirely when the account id or token is absent
    # so gemini/groq stay the chain
    if config.CLOUDFLARE_LLM_BASE_URL and config.CLOUDFLARE_API_TOKEN:
        chain.append(
            openai.LLM(
                model=config.CLOUDFLARE_LLM_MODEL,
                base_url=config.CLOUDFLARE_LLM_BASE_URL,
                api_key=config.CLOUDFLARE_API_TOKEN,
                temperature=config.LLM_TEMPERATURE,
            )
        )
    # Gemini 3.5 Flash follows as fallback; minimal thinking keeps TTFT low
    chain += [
        google.LLM(
            model=model,
            temperature=config.LLM_TEMPERATURE,
            thinking_config={"thinking_level": "minimal"},
            http_options=genai_types.HttpOptions(timeout=config.GEMINI_HTTP_TIMEOUT_MS),
        )
        for model in config.GEMINI_MODELS
    ]
    chain += [
        groq.LLM(model=model, temperature=config.LLM_TEMPERATURE)
        for model in config.GROQ_LLM_MODELS
    ]
    labels = (
        (
            [f"self-hosted:{config.OLLAMA_LLM_MODEL}"]
            if config.SELF_HOSTED_LLM_ENABLED
            else []
        )
        + (
            [config.CLOUDFLARE_LLM_MODEL]
            if config.CLOUDFLARE_LLM_BASE_URL and config.CLOUDFLARE_API_TOKEN
            else []
        )
        + config.GEMINI_MODELS
        + config.GROQ_LLM_MODELS
    )
    logger.info("LLM chain: %s", " -> ".join(labels))
    return HistoryCappedFallback(
        llm=chain, attempt_timeout=config.LLM_ATTEMPT_TIMEOUT_S, sticky=False
    )
