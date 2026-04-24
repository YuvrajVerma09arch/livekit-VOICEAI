"""
Voice Agent with Tool Calling (CAIRA - IT Support)

This script implements a production-grade voice assistant named CAIRA.
It features:
1. Tool Calling: Ability to check weather (via Open-Meteo), server status, and reset passwords.
2. High Availability: Fallback adapters for STT, LLM, and TTS.
3. Intelligence: Semantic Turn Detection to prevent interruptions.
4. Error Handling: Uses ToolError for graceful failure reporting.
"""

import logging
import httpx
import asyncio
from typing import Any
from dotenv import load_dotenv

# Core LiveKit Agent classes
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RunContext,
    TurnHandlingOptions,
    cli,
)

# Modular Agent Capabilities
from livekit.agents import llm, stt, tts, inference
from livekit.agents.llm import function_tool, ToolError

# Provider Plugins
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables containing API keys
load_dotenv()

class Assistant(Agent):
    """
    The Assistant class defines CAIRA's persona and her available tools.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are CAIRA, an experienced Level 2 IT Support Agent for an enterprise service desk. "
                "You are patient, technical but easy to understand, and focused on resolving the user's IT issues efficiently. "
                "You have access to tools to check server statuses, reset passwords, and look up the weather. "
                "Ask clarifying questions if needed, like device type or error codes. "
                "Keep your responses concise, conversational, and natural to speak out loud. "
                "Do not use emojis, markdown formatting, or long lists."
            ),
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> dict[str, Any]:
        """Look up current weather information for the given location."""
        logging.info(f"Tool called: looking up weather for {location}")

        # Explicit business logic error handling
        if location.lower() == "mars":
            raise ToolError("This location is soon coming. Please join our mailing list to stay updated.")

        async with httpx.AsyncClient() as client:
            try:
                # 1. Geocoding: Find coordinates for the location
                geo_response = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                )
                geo_data = geo_response.json()

                if not geo_data.get("results"):
                    raise ToolError(f"Could not find location: {location}")

                lat = geo_data["results"][0]["latitude"]
                lon = geo_data["results"][0]["longitude"]
                place_name = geo_data["results"][0]["name"]

                # 2. Get current weather for those coordinates
                weather_response = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,weather_code",
                        "temperature_unit": "fahrenheit",
                    },
                )
                weather = weather_response.json()

                return {
                    "location": place_name,
                    "temperature_f": weather["current"]["temperature_2m"],
                    "conditions": weather["current"]["weather_code"],
                }
            except httpx.HTTPError as e:
                logging.error(f"HTTP error fetching weather: {e}")
                raise ToolError("The weather service is temporarily unavailable.")
            except Exception as e:
                logging.error(f"Unexpected error in lookup_weather: {e}")
                raise ToolError("An unexpected error occurred while looking up the weather.")

    @function_tool
    async def check_server_status(self, context: RunContext, service_name: str) -> str:
        """Check the current status of a specific corporate server or service."""
        logging.info(f"Tool called: checking status for {service_name}")
        
        if "email" in service_name.lower() or "exchange" in service_name.lower():
            return f"The {service_name} is currently down for maintenance. Estimated recovery: 20 minutes."
        
        return f"The {service_name} is fully operational and reporting no issues."

    @function_tool
    async def reset_password(self, context: RunContext, employee_username: str) -> str:
        """Trigger a password reset link to be sent to an employee's secondary email."""
        logging.info(f"Tool called: resetting password for {employee_username}")
        return f"Done! A secure password reset link has been sent to the recovery email for {employee_username}."

    @function_tool
    async def process_payment(self, context: RunContext, amount: float, account_id: str) -> str:
        """Process a payment for a specific account."""
        
        # Prevent the tool from being cancelled if the user speaks
        context.disallow_interruptions() 
        
        logging.info(f"Tool called: processing payment of ${amount} for account {account_id}. Simulating network delay...")
        
        # Simulate a slow external API call (e.g., reaching out to Stripe or PayPal)
        await asyncio.sleep(5)
        
        return f"Successfully processed the payment of ${amount} for account {account_id}."

    @function_tool()
    async def search_knowledge_base(self, context: RunContext, query: str) -> str:
        """Search the internal IT knowledge base for troubleshooting articles."""
        
        # Send a verbal status update to the user after a short delay
        async def _speak_status_update(delay: float = 0.5):
            await asyncio.sleep(delay)
            await context.session.generate_reply(instructions=f"""
                You are searching the knowledge base for "{query}" but it is taking a little while.
                Update the user on your progress, but be very brief.
            """)
        
        status_update_task = asyncio.create_task(_speak_status_update(0.5))

        logging.info(f"Tool called: searching for '{query}'...")
        
        # Simulate a variable-length search operation
        await asyncio.sleep(8) 
        result = f"Found relevant articles for '{query}'. Suggest rebooting the affected system."
        
        # Cancel status update if search completed before the timeout
        status_update_task.cancel()
        
        return result

# Initialize the Agent Server
server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    Configures the resilient session and starts the agent.
    """
    # 1. Setup resilient Fallback Adapters using the inference factory
    resilient_stt = stt.FallbackAdapter(
        [
            inference.STT.from_model_string("assemblyai/universal-streaming:en"),
            inference.STT.from_model_string("deepgram/nova-3"),
        ]
    )

    resilient_llm = llm.FallbackAdapter(
        [
            inference.LLM(model="openai/gpt-4o-mini"),
            inference.LLM(model="google/gemini-2.0-flash"),
        ]
    )

    resilient_tts = tts.FallbackAdapter(
        [
            inference.TTS.from_model_string(
                "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
            ),
            inference.TTS.from_model_string("inworld/inworld-tts-1"),
        ]
    )

    # 2. Instantiate our Assistant
    assistant = Assistant()

    # 3. Configure the AgentSession
    session = AgentSession(
        stt=resilient_stt,
        llm=resilient_llm,
        tts=resilient_tts,
        vad=silero.VAD.load(),
        # Enabling Semantic Turn Detection
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
    )

    # 4. Start the session with Noise Cancellation
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 5. Establish connection to the room
    await ctx.connect()

if __name__ == "__main__":
    # Setup standard logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the worker app
    cli.run_app(server)