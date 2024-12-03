import os
import sys
import time
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import UpdateUserTyping, UpdateChatUserTyping
from tqdm import tqdm
import asyncio

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.MySQLDB_manager import MySQLDB
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

TARGET_BOT_USERNAME = BBOT_USER

MAX_RETRIES = 1
RESTART_WAIT_TIME = 30  # Time to wait after a crash before retrying

# TARGET_BOT_USERNAME = "@rodolfoccr"

session = os.path.join(TG_SESSION_PATH, "active_session.session")

client = TelegramClient(session, API_ID_TG_RAGTEST, API_HASH_TG_RAGTEST)

def initialize_test_results(test_name):
    """
    Ensures the test_results table exists and inserts missing (test_name, question_id) pairs.
    """
    # Create the test_results table if it doesn't exist
    sql_con.create_table(TEST_RESULTS_TABLE, TEST_RESULTS_SCHEMA)

    # Fetch all active questions
    questions = sql_con.get_records(SQL_EVAL_QAS_TABLE, ["id_question"], {"valid": 1})
    # questions = questions[0:2]

    # Ensure questions are processed as dictionaries
    question_ids = [question["id_question"] if isinstance(question, dict) else question[0] for question in questions]

    # Fetch all existing records for this test
    existing_records = sql_con.get_records(
        TEST_RESULTS_TABLE,
        ["id_question"],
        {"test_name": test_name}
    )
    existing_question_ids = {
        record["id_question"] if isinstance(record, dict) else record[0]
        for record in existing_records
    }

    # Find missing question IDs
    missing_question_ids = set(question_ids) - existing_question_ids

    if missing_question_ids:

        # Prepare records for batch insertion
        records_to_insert = [
            {
                "test_name": test_name,
                "id_question": question_id,
                "status": "pending"
            }
            for question_id in missing_question_ids
        ]

        # Insert all missing records at once
        if records_to_insert:
            sql_con.insert_many_records(TEST_RESULTS_TABLE, records_to_insert)


def fetch_pending_questions(test_name):
    """
    Fetch all questions with status 'pending' for the given test.
    """
    # Retrieve pending questions using the get_records method
    conditions = {"test_name": test_name, "status": "pending"}
    records = sql_con.get_records(
        TEST_RESULTS_TABLE,
        ["id_question"],
        conditions
    )

    if not records:
        return []

    # If records are dictionaries, access with keys
    if isinstance(records[0], dict):
        question_ids = [record["id_question"] for record in records]
    else:  # Otherwise, assume they are tuples and access by index
        question_ids = [record[0] for record in records]

    # Fetch question details for the extracted question IDs
    question_records = sql_con.get_records(
        SQL_EVAL_QAS_TABLE,
        ["id_question", "question"],
        {"id_question": tuple(question_ids)}
    )

    return question_records

def update_question_status(test_name, question_id, status, error_message=None):
    """
    Update the status and error_message of a question in test_results.
    """
    update_data = {"status": status}
    if error_message:
        update_data["error_message"] = error_message
    conditions = {"test_name": test_name, "id_question": question_id}
    sql_con.update_record(TEST_RESULTS_TABLE, update_data, conditions)


async def start_interaction(client):
    """
    Starts interaction with the bot using /start.
    """
    response = await send_message_and_wait(client, TARGET_BOT_USERNAME, "/start", timeout=5)
    if response is None:
        print("Failed to start interaction with /start.")
    return response

async def reset_bot(client):
    """
    Sends /reset to the bot.
    """
    response = await send_message_and_wait(client, TARGET_BOT_USERNAME, "/reset", timeout=5)
    if response is None:
        print("Failed to reset the bot with /reset.")
    return response

async def send_question(client, question_text, timeout=240):
    """
    Sends the question to the bot and waits for the response.
    """
    response = await send_message_and_wait(client, TARGET_BOT_USERNAME, question_text, timeout=timeout, check_typing=True)
    if response is None:
        print("Timeout or no response for question.")
    return response

async def reset_bot(client):
    """
    Sends /reset to the bot and waits for confirmation.
    """
    await asyncio.sleep(0.1)  # Add delay before sending /reset
    response = await send_message_and_wait(client, TARGET_BOT_USERNAME, "/reset", timeout=10)
    if response is None:
        print("Failed to reset the bot: No response received.")
        return False
    if "reset successful" not in response.lower():  # Adjust to match actual reset confirmation text
        print("Failed to reset the bot: Unexpected response:", response)
        return False
    return True

async def handle_crash(client):
    """
    Handles crashes by informing the user, waiting, and restarting the interaction.
    """
    print("Handling crash... Informing user.")
    await notify_user(client, "The test encountered a crash. Attempting to restart...")

    print("Waiting 30 seconds before restarting...")
    await asyncio.sleep(30)

    response = await start_interaction(client)
    if response is None:
        print("No response received after restart. Stopping the test.")
        await notify_user(client, "Test stopped due to no response after restart.")
        return False  # Stop the test
    return True

async def send_message_and_wait(client, chat_id, message, timeout=120, check_typing=False):
    """
    Sends a message to the bot and waits for its response, with optional typing action checks.
    """
    response = None
    typing_detected = False

    async def handle_response(event):
        nonlocal response
        if event.sender_id == (await client.get_input_entity(chat_id)).user_id:
            response = event.raw_text

    async def handle_typing(update):
        nonlocal typing_detected
        if isinstance(update, (UpdateUserTyping, UpdateChatUserTyping)):
            if update.user_id == (await client.get_input_entity(chat_id)).user_id:
                typing_detected = True

    client.add_event_handler(handle_response, events.NewMessage)
    client.add_event_handler(handle_typing, events.Raw)

    try:
        await client.send_message(chat_id, message)
        start_time = time.time()
        while response is None:
            if check_typing and typing_detected:
                typing_detected = False
                start_time = time.time()  # Reset timeout
            if time.time() - start_time > timeout:
                break
            await asyncio.sleep(0.5)
    finally:
        client.remove_event_handler(handle_response, events.NewMessage)
        client.remove_event_handler(handle_typing, events.Raw)

    return response

async def process_question(client, test_name, question):
    """
    Processes a single question by sending it, handling responses, and resetting the bot.
    """
    question_id = question["id_question"]
    question_text = question["question"]

    response = await send_question(client, question_text)
    if response:
        update_question_status(test_name, question_id, "success")
        if not await reset_bot(client):
            update_question_status(test_name, question_id, "error", "Failed to reset after sending question")
            return False
        return True
    else:
        print("No response received. Handling as a crash.")
        if not await handle_crash(client):
            update_question_status(test_name, question_id, "error", "Test stopped due to crash")
            return False

    return False

async def ask_questions(client, test_name):
    """
    Main logic for processing all questions.
    """
    questions = fetch_pending_questions(test_name)
    if not questions:
        print("No pending questions found.")
        return

    if not await start_interaction(client):
        print("Failed to start initial interaction with the bot.")
        return

    with tqdm(total=len(questions), desc="Processing Questions") as progress:
        for question in questions:
            if not await process_question(client, test_name, question):
                print("Stopping test due to repeated failures.")
                break  # Stop test on failure
            progress.update(1)

    print("All questions processed.")

async def send_message_and_wait(client, chat_id, message, timeout=120, check_typing=False):
    """
    Sends a message to the bot and waits for its response, with optional typing action checks.
    """
    response = None
    typing_detected = False

    async def handle_response(event):
        """
        Handles incoming messages from the bot.
        """
        nonlocal response
        if event.sender_id and event.sender_id == (await client.get_input_entity(chat_id)).user_id:
            response = event.raw_text  # Capture the response text

    async def handle_typing(update):
        """
        Detects typing updates from the bot.
        """
        nonlocal typing_detected
        if isinstance(update, (UpdateUserTyping, UpdateChatUserTyping)):
            # Check if the typing update is from the bot
            if hasattr(update, 'user_id') and update.user_id == (await client.get_input_entity(chat_id)).user_id:
                typing_detected = True

    # Event handlers
    client.add_event_handler(handle_response, events.NewMessage)
    client.add_event_handler(handle_typing, events.Raw)

    try:
        await client.send_message(chat_id, message)

        # Wait for response or timeout
        start_time = time.time()
        while response is None:
            if check_typing and typing_detected:
                typing_detected = False
                start_time = time.time()  # Reset timeout start time

            if time.time() - start_time > timeout:
                break
            await asyncio.sleep(0.5)
    finally:
        client.remove_event_handler(handle_response, events.NewMessage)
        client.remove_event_handler(handle_typing, events.Raw)

    return response

async def notify_user(client, message):
    """
    Notifies a user in case of errors or crashes.
    """
    await client.send_message(NOTIFY_USER, message)


async def request_authorization(client, chat_id):
    """
    Handles the authorization process by sending commands to the bot and waiting for responses.
    Sends `/start`, `/getaccess`, and provides a password if requested. Skips steps if authorization is already granted.
    """
    # Start authorization with /start
    response = await send_message_and_wait(client, chat_id, "/start", timeout=10)
    if response is None:
        print("Authorization failed at /start.")
        return False

    # Request access with /getaccess
    response = await send_message_and_wait(client, chat_id, "/getaccess", timeout=10)
    if response is None:
        print("Authorization failed at /getaccess.")
        return False

    # Check if already authorized
    if "du darfst bereits chatten" in response.lower():
        print("Already authorized. Finalizing authorization.")
        response = await send_message_and_wait(client, chat_id, "/end")
        if response is None:
            print("Authorization failed at /end.")
            return False
        print("Authorization successful.")
        return True

    # Provide password if requested
    if "bitte gib das passwort ein" in response.lower():
        password = os.getenv("AUTH_TELEBOT_PWD")  # Retrieve password from environment
        response = await send_message_and_wait(client, chat_id, password, timeout=10)
        if response is None:
            print("Authorization failed at password prompt.")
            return False

    # Finalize authorization
    response = await send_message_and_wait(client, chat_id, "/end", timeout=10)
    if response is None:
        print("Authorization failed at /end.")
        return False

    print("Authorization successful.")
    return True

if __name__ == "__main__":
    # test_name = "Test Run Manual 4"
    test_name = sys.argv[1] if len(sys.argv) > 1 else "Default Test"

    async def main():
        async with client:
            # Request authorization and start interaction
            authorized = await request_authorization(client, TARGET_BOT_USERNAME)
            if not authorized:
                print("Failed to obtain authorization. Exiting...")
                return

            # Send /start after authorization
            if not await start_interaction(client):
                print("Failed to start interaction after authorization. Exiting...")
                return

            # Initialize test results and start processing questions
            initialize_test_results(test_name)
            await ask_questions(client, test_name)

    asyncio.run(main())