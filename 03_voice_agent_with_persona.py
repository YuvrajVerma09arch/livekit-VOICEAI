"""
Voice Agent with Persona (CAIRA - IT Support)

This script sets up a real-time LiveKit voice assistant using the LiveKit Agents framework.
It configures specific STT (Deepgram), LLM (OpenAI), and TTS (Cartesia) plugins 
to create a custom, highly-responsive persona named CAIRA, an enterprise IT Support Agent.
"""

import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables (e.g., API keys for LiveKit, OpenAI, Deepgram, Cartesia)
load_dotenv()

class Assistant(Agent):
    """
    The Assistant class defines the core personality and conversational behavior of the voice AI.

    By passing specific instructions (System Prompt) to the parent Agent class, 
    we shape how the LLM interprets queries and formats its responses to act as 
    a specialized persona—in this case, a Level 2 IT Support Agent.
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

# Initialize the Agent Server to manage incoming worker connections and RTC sessions
server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    The main entrypoint for the LiveKit agent worker. 
    This function is automatically triggered whenever the worker connects to a LiveKit room.

    Args:
        ctx (JobContext): The context object providing access to the current room, 
                          participants, and connection state.
    """
    # 1. Configure the AgentSession with our chosen AI models and plugins
    session = AgentSession(
        # Customizing Speech-to-Text (STT) with Deepgram
        stt="deepgram/nova-3",
        
        # Setting the Large Language Model (LLM) for dialogue generation
        llm="openai/gpt-4.1-mini",
        
        # Customizing Text-to-Speech (TTS) with Cartesia to give the agent a unique, consistent voice
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        
        # Loading the Silero Voice Activity Detection (VAD) model to detect user speech boundaries
        vad=silero.VAD.load(),
        
        # Configuring Semantic Turn Detection to wait for natural sentence completion before interrupting
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
        ),
    )

    # 2. Start the session, binding the Assistant persona and configuring room-level audio options
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # Apply Background Voice Cancellation (BVC) to prevent the agent from hearing and reacting to its own echo
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 3. Establish the final connection to the LiveKit room to begin the interaction
    await ctx.connect()

if __name__ == "__main__":
    # Setup standard logging to output agent status and errors to the console
    logging.basicConfig(level=logging.INFO)
    
    # Boot up the worker app using the configured AgentServer instance
    cli.run_app(server)