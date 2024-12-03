import os
import sys
import time
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import UpdateUserTyping, UpdateChatUserTyping, InputPeerUser
from telethon.tl.functions.messages import SendMessageRequest
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
messages_to_send = [
    "Hello Bot, this is message 1.",
    "This is message 2.",
    "And this is message 3."
]

MAX_RETRIES = 1
RESTART_WAIT_TIME = 30  # Time to wait after a crash before retrying

# TARGET_BOT_USERNAME = "@rodolfoccr"

session = os.path.join(TG_SESSION_PATH, "active_session.session")

client = TelegramClient(session, API_ID_TG_RAGTEST, API_HASH_TG_RAGTEST)

async def handle_messages(bot_username, admin_username, messages_to_send):
    try:
        async with client.conversation(bot_username) as conv:
            for message in messages_to_send:
                print(f"Sending message: {message}")

                # Step 1: Send the message
                await conv.send_message(message)

                # Step 2: Wait for bot's response
                while True:
                    bot_response = await conv.get_response()
                    print(f"Bot sent: {bot_response.text}")

                    # Check for crash message
                    if "crash" in bot_response.text.lower():
                        print("Bot reported a crash.")
                        await client.send_message(admin_username, "Bot reported a crash during interaction.")
                        # Stop processing further messages
                        return

                    # Check for confirmation question
                    if "war das eine gute antwort?" in bot_response.text.lower():
                        print("Received confirmation question.")
                        # Wait and confirm
                        await asyncio.sleep(0.4)
                        await conv.send_message("/reset")
                        break  # Exit the while loop to move to the next message

                # Step 3: Wait for two additional messages
                additional_messages = []
                for _ in range(2):
                    bot_additional_response = await conv.get_response()
                    print(f"Bot sent additional message: {bot_additional_response.text}")
                    additional_messages.append(bot_additional_response.text)

                # Step 4: Evaluate the second additional message
                prompt_id,device = additional_messages[1].split("|")
                print(f"Prompt id: {prompt_id} Device: {device}")
                # Perform your evaluation or actions here

                print(f"Completed processing message: {message}")

    except Exception as e:
        # Handle unexpected failures
        print(f"Unexpected failure during interaction: {e}")
        await client.send_message(admin_username, "Unexpected failure during interaction with the bot.")
        # Perform additional actions if required
    finally:
        print("Interaction complete.")

async def main():
    await handle_messages(bot_username, admin_username, messages_to_send)

with client:
    client.loop.run_until_complete(main())