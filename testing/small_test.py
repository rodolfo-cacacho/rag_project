import os
import sys
import time
import asyncio
from telethon import TelegramClient, events
from dotenv import load_dotenv

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (TG_SESSION_PATH)

# Load environment variables
load_dotenv()

API_HASH_TG_RAGTEST = os.getenv('API_HASH_TG_RAGTEST')
API_ID_TG_RAGTEST = os.getenv('API_ID_TG_RAGTEST')

TARGET_BOT_USERNAME = "@rodolfoccr"  # Replace with your target bot's username

session = os.path.join(TG_SESSION_PATH, "test_session.session")

client = TelegramClient(session, API_ID_TG_RAGTEST, API_HASH_TG_RAGTEST)

async def test_message_interaction():
    """
    Send a message to the bot, await the response, and reply after a second.
    """
    async def handle_response(event):
        """
        Handles incoming messages from the bot.
        """
        print(f"New message received: {event.raw_text}")  # Log all messages for debugging

        # Check if the event has a valid sender ID and matches the target bot
        if event.sender_id and event.sender_id == await client.get_peer_id(TARGET_BOT_USERNAME):
            print(f"Received response from target bot: {event.raw_text}")

            # Wait for a second and reply back
            await asyncio.sleep(1)
            await client.send_message(event.chat_id, "Replying back after a second.")
        else:
            print("Message received from an unknown sender.")

    # Add event handler for new messages
    client.add_event_handler(handle_response, events.NewMessage)

    try:
        # Start the interaction
        print("Sending a test message...")
        await client.send_message(TARGET_BOT_USERNAME, "Test message from client.")

        # Keep the script running to listen for responses
        print("Waiting for a response. Press Ctrl+C to stop.")
        await asyncio.sleep(60)  # Keep the script alive for a minute for testing
    finally:
        client.remove_event_handler(handle_response, events.NewMessage)


if __name__ == "__main__":
    async def main():
        async with client:
            await test_message_interaction()

    asyncio.run(main())