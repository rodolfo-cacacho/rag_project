import os
import sys
import time
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import UpdateUserTyping, UpdateChatUserTyping, InputPeerUser
from telethon.tl.functions.messages import SendMessageRequest
from telethon.tl.types import UpdateShortChatMessage, UpdateShortMessage
from typing import List, Callable, Awaitable
from tqdm import tqdm
import asyncio

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.MySQLDB_manager import MySQLDB
from testing.modules.testing_chat import initialize_test_results,fetch_pending_questions,update_question_status
from config import (
    CONFIG_SQL_DB,
    DB_NAME,
    TG_SESSION_PATH,
    BBOT_USER,
    TEST_RESULTS_TABLE,
    TEST_RESULTS_SCHEMA,
    SQL_EVAL_QAS_TABLE,
    NOTIFY_USER
)

# Load environment variables
load_dotenv()

sql_con = MySQLDB(CONFIG_SQL_DB, DB_NAME)

API_HASH_TG_RAGTEST = os.getenv('API_HASH_TG_RAGTEST')
API_ID_TG_RAGTEST = os.getenv('API_ID_TG_RAGTEST')

bot_username = "@rodolfocco"
admin_username = "@rodolfocco"
questions = [
    "Hello Bot, this is message 1.",
    "This is message 2.",
    "And this is message 3."
]

password = "YOUR_PASSWORD"

MAX_RETRIES = 1
RESTART_WAIT_TIME = 30  # Time to wait after a crash before retrying

# TARGET_BOT_USERNAME = "@rodolfoccr"

session = os.path.join(TG_SESSION_PATH, "active_session.session")
async def handle_bot_conversation(
    client, 
    bot_username: str, 
    messages: List[str], 
    expected_response_checker: Callable[[str], bool],
    crash_indicator: str = "Error",
    reset_command: str = "/reset",
    max_wait_time: float = 60.0,
    crash_wait_time: float = 10.0
):
    """
    Handle a conversation with a Telegram bot using Telethon.
    """
    bot_responses = []
    
    # Create an event to signal when a response is received
    response_event = asyncio.Event()
    current_message = None
    
    @client.on(events.NewMessage(from_users=bot_username))
    async def message_handler(event):
        nonlocal current_message, bot_responses
        
        # Check for crash
        if crash_indicator in event.raw_text:
            print(f"Bot crashed. Waiting {crash_wait_time} seconds.")
            await asyncio.sleep(crash_wait_time)
            response_event.set()
            return
        
        # Check if response is valid for the current message
        if current_message is not None and expected_response_checker(event.raw_text):
            bot_responses.append(event.raw_text)
            response_event.set()
    
    for message in messages:
        # Reset event
        response_event.clear()
        
        # Set current message context
        current_message = message
        
        # Send message to bot
        await client.send_message(bot_username, message)
        
        # Wait for response or timeout
        try:
            await asyncio.wait_for(response_event.wait(), timeout=max_wait_time)
        except asyncio.TimeoutError:
            print(f"No valid response received for message: {message}")
            continue
        
        # Send reset command
        await client.send_message(bot_username, reset_command)
        
        # Short wait to ensure reset is processed
        await asyncio.sleep(1)
    
    # Remove the event handler
    client.remove_event_handler(message_handler)
    
    return bot_responses

async def main():
    # Initialize your Telethon client
    client = TelegramClient(session, API_ID_TG_RAGTEST, API_HASH_TG_RAGTEST)
    
    try:
        # Connect to the client
        await client.connect()
               
        # Define your messages
        messages = [
            "Hello, how are you?", 
            "What's your favorite color?"
        ]
        
        # Define a response checker function
        def check_response(response):
            # Example: Check if response contains certain keywords
            return "es war gut" in response.lower() or "super" in response.lower()
        
        # Run the conversation handler
        responses = await handle_bot_conversation(
            client, 
            bot_username, 
            messages, 
            check_response
        )
        
        print("Bot Responses:", responses)
    
    finally:
        # Ensure client is disconnected
        await client.disconnect()

if __name__ == "__main__":
    # test_name = "Test Run Manual 4"
    test_name = sys.argv[1] if len(sys.argv) > 1 else "Default Test"
    asyncio.run(main())