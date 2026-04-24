import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    cli,
)
from livekit.plugins import noise_cancellation, silero

# Load environment variables containing API keys
load_dotenv()


class Assistant(Agent):
    """
    The Assistant class defines the core behavior and system prompt 
    for our voice AI.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant.",
        )


# --- Step 3: Configuring the Agent Server ---
# Now we instantiate our `AgentServer`. This object is responsible for managing 
# the incoming agent workers and their sessions.

# Initialize the Agent Server
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    The entrypoint is triggered when the worker connects to a LiveKit room.
    """
    # 1. Configure the AI pipeline models
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2",
        vad=silero.VAD.load(),
    )

    # --- Step 5: Starting the Session and Connecting ---
    # Once the session is configured, we need to start it by passing in our `Assistant` 
    # instance, the LiveKit room object, and any input options.
    #
    # To ensure the AI doesn't get confused by background noise or its own voice echoing, 
    # we apply Background Voice Cancellation (BVC) using the `noise_cancellation` plugin. 
    # Finally, we establish the connection to the room.

    # 2. Start the session with our Assistant and Room options
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 3. Connect the agent to the room
    await ctx.connect()



if __name__ == "__main__":
    # Setup standard logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the worker app using the AgentServer
    cli.run_app(server)