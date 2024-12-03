import os
import sys
# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from pyrogram import Client, filters,idle
from pyrogram.types import Message
from time import sleep
from dotenv import load_dotenv
from enum import Enum, auto
import asyncio

# Enum for State Management
class ChatState(Enum):
    START = auto()
    REQUEST = auto()
    AUTHORIZED = auto()
    PWD = auto()
    AUTH_READY = auto()
    ASK = auto()
    WAIT_RESTART = auto()
    POST_ASK = auto()
    END = auto()

load_dotenv()

API_HASH_TG_RAGTEST = os.getenv('API_HASH_TG_RAGTEST')
API_ID_TG_RAGTEST = os.getenv('API_ID_TG_RAGTEST')

# Initialize Pyrogram client for a user account
app = Client(
    "my_account",
    api_id=API_ID_TG_RAGTEST,
    api_hash=API_HASH_TG_RAGTEST
)

bot_username = "@rodolfocco"

# Chat state data
chat_states = {}

def set_state(chat_id, new_state):
    """Set the state for a specific chat."""
    chat_states[chat_id] = new_state

def get_state(chat_id):
    """Get the state for a specific chat."""
    return chat_states.get(chat_id, ChatState.START)

@app.on_message(filters.private)
def handle_messages(client, message):
    """
    Handles incoming messages in a stateful manner.
    """
    chat_id = message.chat.id
    current_state = get_state(chat_id)
    print(f"Received message: {message.text}, State: {current_state}, Chat ID: {chat_id}")

    if current_state == ChatState.START:
        app.send_message(chat_id, "Hi! Let's start! What's your name?")
        set_state(chat_id, ChatState.QUESTION_1)

    elif current_state == ChatState.QUESTION_1:
        user_name = message.text
        app.send_message(chat_id, f"Nice to meet you, {user_name}! What is your favorite color?")
        set_state(chat_id, ChatState.QUESTION_2)

    elif current_state == ChatState.QUESTION_2:
        favorite_color = message.text
        app.send_message(chat_id, f"{favorite_color} is a beautiful color! Thank you for chatting.")
        set_state(chat_id, ChatState.END)

    elif current_state == ChatState.END:
        app.send_message(chat_id, "The conversation has ended. Type /start to restart.")
        set_state(chat_id, ChatState.START)

async def send_start_message():
    """
    Sends the /start message to the bot programmatically.
    """
    print(f"Sending /start to {bot_username}...")
    await app.send_message(bot_username, "/start")
    print("Initial /start message sent.")

async def main():
    """
    Starts the client, sends the initial /start message, and runs the bot.
    """
    await app.start()  # Start the client
    await send_start_message()  # Send the initial /start message
    print("Bot is running...")
    await idle()  # Keep the bot running to process incoming messages

if __name__ == "__main__":
    asyncio.run(main())  # Run the main function