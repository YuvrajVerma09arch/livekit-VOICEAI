"""Function tools exposed to the model.

Standalone `@function_tool` functions passed to the agent via `tools=[...]`,
so tool logic never mixes with agent/session wiring.
"""

import asyncio
import logging
from typing import Any

import httpx
from ddgs import DDGS
from livekit.agents.llm import ToolError, function_tool

from . import config

logger = logging.getLogger(config.LOG_NAME)

# WMO weather interpretation codes, https://open-meteo.com/en/docs
_WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


@function_tool
async def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Returns the top results as title, url and
    a short snippet. Use this whenever a fact may have changed or you're unsure."""

    def _search() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return ddgs.text(query, max_results=config.WEB_SEARCH_RESULTS)

    # ddgs is blocking, so run it off the event loop to keep audio flowing;
    # the wait_for bounds a hung DuckDuckGo call so a turn fails fast instead
    # of stalling until the default socket timeout
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_search), timeout=config.WEB_SEARCH_TIMEOUT_S
        )
    except TimeoutError:
        logger.warning(f"web search timed out after {config.WEB_SEARCH_TIMEOUT_S}s: {query!r}")
        raise ToolError("The web search timed out, please ask again.")
    except Exception:
        logger.exception("web search failed")
        raise ToolError("The web search is unavailable right now, please try again.")

    if not results:
        return f"No results found for {query}."

    return "\n".join(
        f"{n}. {r.get('title', '')} ({r.get('href', '')}): {r.get('body', '')}"
        for n, r in enumerate(results, start=1)
    )


@function_tool
async def lookup_weather(location: str) -> dict[str, Any]:
    """Look up the current weather for a location as {location, conditions, temperature_f}."""
    logger.info(f"Tool called: looking up weather for {location}")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
        except httpx.HTTPError:
            logger.exception("geocoding request failed")
            raise ToolError("The geocoding service is temporarily unavailable.")
        except (ValueError, KeyError, IndexError):
            logger.exception("unexpected geocoding payload")
            raise ToolError("Could not find that location.")

        try:
            features = geo_data["results"]
            place = features[0]
            latitude = place["latitude"]
            longitude = place["longitude"]
        except (KeyError, TypeError, IndexError):
            logger.exception(f"no geocoding results for {location!r}")
            raise ToolError(f"Could not find a place called {location}.")

        try:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,weather_code",
                    "temperature_unit": "fahrenheit",
                },
            )
            response.raise_for_status()
            weather = response.json()["current"]
        except httpx.HTTPError:
            logger.exception("open-meteo request failed")
            raise ToolError("The weather service is temporarily unavailable.")
        except (ValueError, KeyError, TypeError):
            logger.exception("unexpected open-meteo payload")
            raise ToolError("An unexpected error occurred while fetching the weather.")

        return {
            "location": place.get("name", location),
            "temperature_f": weather["temperature_2m"],
            "conditions": _WMO_CODES.get(weather["weather_code"], "unknown conditions"),
        }


AGENT_TOOLS = [web_search, lookup_weather]
