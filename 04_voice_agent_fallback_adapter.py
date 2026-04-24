"""
Voice Agent with Fallback Adapters

This script sets up a highly available LiveKit voice assistant.
It configures Fallback Adapters for STT, LLM, and TTS pipelines to ensure 
that if a primary AI provider fails or rate-limits, the assistant automatically 
and seamlessly switches to a backup provider.
"""

import logging
from dotenv import load_dotenv

# Core LiveKit Agent classes
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    TurnHandlingOptions,
    cli,
)

# Modular Agent Capabilities for routing and factories
from livekit.agents import llm, stt, tts, inference

# Provider Plugins (for utilities like VAD, Noise Cancellation, and Turn Detection)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables containing API keys for all providers
load_dotenv()

class Assistant(Agent):
    """
    The Assistant class defines the core personality and conversational behavior.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are CAIRA, an experienced Level 2 IT Support Agent for an enterprise service desk. "
                "You are patient, technical but easy to understand, and focused on resolving the user's IT issues efficiently. "
                "Ask clarifying questions if needed, like device type or error codes. "
                "Keep your responses concise, conversational, and natural to speak out loud. "
                "Do not use emojis, markdown formatting, or long lists."
            ),
        )

# Initialize the Agent Server to manage incoming worker connections
server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    The main entrypoint for the LiveKit agent worker.
    """
    
    # 1. Define STT Fallback (AssemblyAI -> Deepgram)
    resilient_stt = stt.FallbackAdapter(
        [
            inference.STT.from_model_string("assemblyai/universal-streaming:en"),
            inference.STT.from_model_string("deepgram/nova-3"),
        ]
    )

    # 2. Define LLM Fallback (OpenAI -> Google Gemini)
    resilient_llm = llm.FallbackAdapter(
        [
            inference.LLM(model="openai/gpt-4.1-mini"),
            inference.LLM(model="google/gemini-2.5-flash"),
        ]
    )

    # 3. Define TTS Fallback (Cartesia -> Inworld)
    resilient_tts = tts.FallbackAdapter(
        [
            inference.TTS.from_model_string(
                "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
            ),
            inference.TTS.from_model_string("inworld/inworld-tts-1"),
        ]
    )
    
    # 4. Inject the resilient adapters and Turn Detection into the AgentSession
    session = AgentSession(
        stt=resilient_stt,
        llm=resilient_llm,
        tts=resilient_tts,
        vad=silero.VAD.load(),
        # Configuring Semantic Turn Detection to wait for natural sentence completion before interrupting
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
    )

    # 5. Start the session and configure room-level audio options
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 6. Establish the final connection to the LiveKit room
    await ctx.connect()

if __name__ == "__main__":
    # Setup standard logging to output agent status and fallback events
    logging.basicConfig(level=logging.INFO)
    
    # Boot up the worker app using the configured AgentServer instance
    cli.run_app(server)