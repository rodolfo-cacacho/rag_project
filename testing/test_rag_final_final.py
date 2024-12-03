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

password = os.getenv("AUTH_TELEBOT_PWD")


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


async def authorize_user(bot_username, password, timeout=5):
    try:
        async with client.conversation(bot_username) as conv:
            print("Starting interaction with /start")
            await conv.send_message("/start")
            start_response = await conv.get_response(timeout=timeout)
            print(f"Bot response to /start: {start_response.text}")

            print("Requesting authorization with /getaccess")
            await conv.send_message("/getaccess")
            access_response = await conv.get_response(timeout=timeout)
            print(f"Bot response to /getaccess: {access_response.text}")

            if "permission granted" in access_response.text.lower():
                print("Permission already granted. Sending /end.")
                await conv.send_message("/end")
                end_response = await conv.get_response(timeout=timeout)
                print(f"Bot response to /end: {end_response.text}")
                return True

            print("Access not granted. Sending password.")
            await conv.send_message(password)
            password_response = await conv.get_response(timeout=timeout)
            print(f"Bot response to password: {password_response.text}")

            if "password correct" in password_response.text.lower():
                print("Password accepted. Sending /end.")
                await conv.send_message("/end")
                end_response = await conv.get_response(timeout=timeout)
                print(f"Bot response to /end: {end_response.text}")
                return True
            else:
                print("Password incorrect. Authorization failed.")
                return False

    except asyncio.TimeoutError:
        print(f"No response from bot within {timeout} seconds. Authorization failed.")
        return False
    except Exception as e:
        print(f"Unexpected failure during authorization: {e}")
        return False
    finally:
        print("Authorization process complete.")



if __name__ == "__main__":
    # test_name = "Test Run Manual 4"
    test_name = sys.argv[1] if len(sys.argv) > 1 else "Default Test"

    async def main():

        print("Starting authorization process...")
        is_authorized = await authorize_user(bot_username, password)

        if is_authorized:
            print("Authorization successful. Proceeding to ask questions.")
            print("Load questions to ask")
            initialize_test_results(sql_con=sql_con,test_name=test_name,
                                    test_results_table=TEST_RESULTS_TABLE,
                                    test_results_table_schema=TEST_RESULTS_SCHEMA,
                                    sql_eval_table=SQL_EVAL_QAS_TABLE)
            questions = fetch_pending_questions(sql_con=sql_con,
                                    test_name=test_name,
                                    test_table=TEST_RESULTS_TABLE,
                                    qas_table=SQL_EVAL_QAS_TABLE)
            
            # TRIMMING QUESTIONS
            questions = questions[0:2]

            print(f"Questions: {len(questions)}")


            # success = await ask_questions(bot_username, admin_username, questions)
            # if success:
            #     print("All questions processed successfully.")
            # else:
            #     print("Failed to process all questions.")
        else:
            print("Authorization failed. Unable to proceed.")

        # await handle_messages(bot_username, admin_username, messages_to_send)

    with client:
        client.loop.run_until_complete(main())