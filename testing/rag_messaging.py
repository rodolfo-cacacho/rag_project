import os
import sys
# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from pyrogram import Client, filters
from pyrogram.types import Message
from dotenv import load_dotenv
from enum import Enum, auto
from utils.MySQLDB_manager import MySQLDB
from testing.modules.testing_chat import initialize_test_results,fetch_pending_questions,update_question_status
import re
from config import (
    CONFIG_SQL_DB,
    DB_NAME,
    TG_SESSION_PATH,
    BBOT_USER,
    TEST_RESULTS_TABLE,
    TEST_RESULTS_SCHEMA,
    SQL_EVAL_QAS_TABLE,
    NOTIFY_USER,
)

from tqdm import tqdm

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

load_dotenv()

API_HASH_TG_RAGTEST = os.getenv('API_HASH_TG_RAGTEST')
API_ID_TG_RAGTEST = os.getenv('API_ID_TG_RAGTEST')

# Enum for State Management
class ChatState(Enum):
    START_M = auto()
    START = auto()
    REQUEST = auto()
    AUTHORIZED = auto()
    PWD = auto()
    AUTH_READY = auto()
    ASK = auto()
    WAIT_RESTART = auto()
    POST_ASK = auto()
    END = auto()

class QuestionFetcher:
    def __init__(self, questions):
        self.questions = questions
        self.current_index = 0

    def get_next_question(self):
        """Fetch the next question and advance the index."""
        if self.current_index+1 < len(self.questions):
            self.current_index += 1
            question = self.questions[self.current_index]
            return question
        else:
            return None  # No more questions

    def get_current_question(self):
        """Fetch the current question without advancing the index."""
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        else:
            return None  # No current question available

# Initialize Pyrogram client for a user account
app = Client(
    "my_account",
    api_id=API_ID_TG_RAGTEST,
    api_hash=API_HASH_TG_RAGTEST
)

bot_username = BBOT_USER
admin_username = NOTIFY_USER

password = os.getenv("AUTH_TELEBOT_PWD")
# Chat state data
chat_states = {}
fetchers = {}  # Store fetchers globally
progress_bars = {}  # Store progress bars globally

def set_state(chat_id, new_state):
    """Set the state for a specific chat."""
    chat_states[chat_id] = new_state

def get_state(chat_id):
    """Get the state for a specific chat."""
    return chat_states.get(chat_id, ChatState.START_M)

def get_or_create_fetcher(chat_id, questions=None):
    """Get an existing fetcher or create a new one."""
    if chat_id not in fetchers:
        fetchers[chat_id] = QuestionFetcher(questions)
    return fetchers[chat_id]

def create_progress_bar(chat_id, total_steps, description="Progress"):
    """Initialize a progress bar for the chat."""
    if chat_id not in progress_bars:
        progress_bars[chat_id] = tqdm(total=total_steps, desc=description, unit="step")

def update_progress_bar(chat_id, steps=1):
    """Update the progress bar for the chat."""
    if chat_id in progress_bars:
        progress_bars[chat_id].update(steps)

def close_progress_bar(chat_id):
    """Close the progress bar for the chat."""
    if chat_id in progress_bars:
        progress_bars[chat_id].close()
        del progress_bars[chat_id]

@app.on_message(filters.private)
def handle_messages(client, message):
    """
    Handles incoming messages in a stateful manner.
    """
    key = "UNIQUE"
    chat_id = message.chat.id
    current_state = get_state(key)
    print(f"Received message: {message.text}, State: {current_state}, Chat ID: {chat_id}")
    message_text = message.text.lower()

    if current_state == ChatState.START_M:
        app.send_message(bot_username, "/start")
        set_state(key, ChatState.START)

    elif current_state == ChatState.START:
        if 'willkommen' in message_text:
            # ACTION
            app.send_message(bot_username,"/getaccess")
            # SET NEW STATE
            set_state(key,ChatState.REQUEST)
        elif 'Das RAG-System ist abgestürzt'.lower() in message_text:
            # ACTION
            message_send = f"RAG Crashed in {current_state}"
            app.send_message(admin_username,message_send)
            # SET NEW STATE
            set_state(key,ChatState.END)

    elif current_state == ChatState.REQUEST:
        if 'Du darfst nicht chatten'.lower() in message_text:
            # ACTION
            message_send = password
            app.send_message(bot_username,message_send)
            # SET NEW STATE
            set_state(key,ChatState.PWD)

        elif 'Du darfst bereits chatten'.lower() in message_text:
            # ACTION
            message_send = f"/end"
            app.send_message(bot_username,message_send)
            # SET NEW STATE
            set_state(key,ChatState.AUTHORIZED)

        elif 'Das RAG-System ist abgestürzt'.lower() in message_text:
            # ACTION
            message_send = f"RAG Crashed in {current_state}"
            app.send_message(admin_username,message_send)
            # END
            set_state(key,ChatState.END)

    elif current_state == ChatState.PWD:
        if 'Richtiges Passwort'.lower() in message_text:
            # ACTION
            message_send = f'/end'
            app.send_message(bot_username,message_send)
            # SET NEW STATE
            set_state(key,ChatState.AUTHORIZED)

        elif 'Falsches Passwort'.lower() in message_text:
            # ACTION
            message_send = f"Wrong Password - Fail to Authorize User"
            app.send_message(admin_username,message_send)
            # SET NEW STATE
            set_state(key,ChatState.END)

        elif 'Das RAG-System ist abgestürzt'.lower() in message_text:
            # ACTION
            message_send = f"RAG Crashed in {current_state}"
            app.send_message(admin_username,message_send)
            # END
            set_state(key,ChatState.END)

    elif current_state == ChatState.AUTHORIZED:
        if 'Es war mir eine Freude, dir zu helfen'.lower() in message_text:
            # ACTION
            initialize_test_results(sql_con=sql_con,
                                    test_name= test_name,
                                    test_results_table=TEST_RESULTS_TABLE,
                                    test_results_table_schema=TEST_RESULTS_SCHEMA,
                                    sql_eval_table=SQL_EVAL_QAS_TABLE)
            
            questions = fetch_pending_questions(sql_con=sql_con,
                                                test_name=test_name,
                                                test_table=TEST_RESULTS_TABLE,
                                                qas_table=SQL_EVAL_QAS_TABLE)

            if len(questions) > 0:

                fetcher = get_or_create_fetcher(key,questions)

                message_send = f'TIME TO ASK {len(questions)} QUESTIONS'
                app.send_message(admin_username,message_send)

                next_question = fetcher.get_current_question()
                create_progress_bar(key,len(questions),description="Asking Questions")
                               
                app.send_message(bot_username,next_question['question'])
                # SET NEW STATE
                set_state(key,ChatState.ASK)
            else:
                app.send_message(bot_username,"No questions to ask.")
                set_state(key,ChatState.END)


        elif 'Das RAG-System ist abgestürzt'.lower() in message_text:
            # ACTION
            message_send = f"RAG Crashed in {current_state}"
            app.send_message(admin_username,message_send)
            # END
            set_state(key,ChatState.END)

    elif current_state == ChatState.ASK:
        if 'War das eine gute Antwort?'.lower() in message_text:
            # ACTION
            message_send = f'/reset'
            app.send_message(bot_username,message_send)
            # SET NEW STATE
            set_state(key,ChatState.POST_ASK)
        elif 'Das RAG-System ist abgestürzt'.lower() in message_text:
            # ACTION
            fetcher = get_or_create_fetcher(key)
            current_question = fetcher.get_current_question()
            message_send = f"RAG Crashed in {current_state} @ Question {current_question['id_question']} {current_question['question']}"
            app.send_message(admin_username,message_send)
            update_question_status(status='error',
                                   sql_con=sql_con,
                                   test_name=test_name,
                                   question_id=current_question['id_question'],
                                   test_table=TEST_RESULTS_TABLE,
                                   prompt_id=None,
                                   device=None,
                                   error_message="Unknown Error")
            set_state(key,ChatState.WAIT_RESTART)

    elif current_state == ChatState.POST_ASK:
        if re.match(r"^\d+\|.*$", message_text):
            prompt_id,device = message_text.split("|")
            fetcher = get_or_create_fetcher(key)
            current_question = fetcher.get_current_question()
            update_question_status(status='success',
                                   sql_con=sql_con,
                                   test_name=test_name,
                                   question_id=current_question['id_question'],
                                   test_table=TEST_RESULTS_TABLE,
                                   prompt_id=prompt_id,
                                   device=device)
            update_progress_bar(key)
            next_question = fetcher.get_next_question()
            if next_question:
                app.send_message(bot_username,next_question['question'])
                set_state(key,ChatState.ASK)
            else:
                close_progress_bar(key)
                app.send_message(admin_username,"Asked all questions")
                set_state(key,ChatState.END)

                
        elif 'Das RAG-System ist abgestürzt'.lower() in message_text:
            # ACTION
            fetcher = get_or_create_fetcher(key)
            current_question = fetcher.get_current_question()
            message_send = f"RAG Crashed in {current_state} @ Question {current_question['id_question']} {current_question['question']}"
            app.send_message(admin_username,message_send)
            update_question_status(status='error',
                                   sql_con=sql_con,
                                   test_name=test_name,
                                   question_id=current_question['id_question'],
                                   test_table=TEST_RESULTS_TABLE,
                                   prompt_id=None,
                                   device=None,
                                   error_message="Unknown Error")
            # END
            set_state(key,ChatState.WAIT_RESTART)
        
    elif current_state == ChatState.WAIT_RESTART:
        if "RAG-System ist wieder online".lower() in message_text:
            update_progress_bar(key)
            fetcher = get_or_create_fetcher(key)
            next_question = fetcher.get_next_question()
            if next_question:
                app.send_message(bot_username,next_question['question'])
                set_state(key,ChatState.ASK)
            else:
                close_progress_bar(key)
                app.send_message(admin_username,"Asked all questions")
                set_state(key,ChatState.END)                
            
    elif current_state == ChatState.END:
        app.send_message(admin_username, "The conversation has ended. Type /start to restart.")
        set_state(key, ChatState.START_M)
# Pyrogram client setup

# Start the bot
if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "DefaultTest"
    app.run()