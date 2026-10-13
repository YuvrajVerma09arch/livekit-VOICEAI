"""`python -m backend.agent <cmd>` - thin wrapper around the LiveKit CLI."""

from livekit.agents import cli

from .server import server

if __name__ == "__main__":
    cli.run_app(server)
