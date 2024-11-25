
import telebot
from datetime import datetime
from threading import Timer
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the bot with your token
API_KEY_TG_NOTIFY = os.getenv('API_KEY_TG_NOTIFY')

bot = telebot.TeleBot(API_KEY_TG_NOTIFY)

# Define the chat ID for sending notifications
admin_chat_id = os.getenv('ADMIN_TG_ID')

# Import the database connection class
from utils.MySQLDB_manager import MySQLDB
from config import (CONFIG_SQL_DB,DB_NAME)

sql_db_connector = MySQLDB(CONFIG_SQL_DB, DB_NAME)

# Track the last checked message time
last_message_time = {}

def initialize_last_message_time():
    """Initialize the last_message_time dictionary to avoid sending notifications on the first run."""
    global last_message_time
    try:
        # Fetch the most recent messages to initialize the last_message_time dictionary
        query = """
        SELECT m.chat_id, MAX(m.date) as last_message_time
        FROM messages m
        GROUP BY m.chat_id
        """
        sql_db_connector.cursor.execute(f"USE {DB_NAME}")
        sql_db_connector.cursor.execute(query)
        messages = sql_db_connector.cursor.fetchall()

        # Initialize last_message_time for each chat_id
        for chat_id, last_time in messages:
            last_message_time[chat_id] = last_time

    except Exception as e:
        bot.send_message(admin_chat_id, f"Error occurred during initialization: {str(e)}")

def check_for_new_messages():
    global last_message_time
    
    try:
        # Query to fetch the latest messages from the database
        query = """
        SELECT m.chat_id, u.name, m.message, m.date
        FROM messages m
        JOIN user_db u ON m.chat_id = u.id
        ORDER BY m.date DESC
        LIMIT 10  -- Limit to the last 10 messages
        """
        sql_db_connector.cursor.execute(f"USE {DB_NAME}")
        sql_db_connector.cursor.execute(query)
        messages = sql_db_connector.cursor.fetchall()
        
        # Iterate through the fetched messages
        for message_row in messages:
            chat_id, username, content, date = message_row

            # If this is the first time checking or a new message is detected
            if chat_id not in last_message_time or date > last_message_time[chat_id]:
                # Send notification to the admin
                notification_message = f"User {username} (Chat ID: {chat_id}) sent a message: {content} at {date}"
                bot.send_message(admin_chat_id, notification_message)
                
                # Update the last message time for this user
                last_message_time[chat_id] = date
                
    except Exception as e:
        bot.send_message(admin_chat_id, f"Error occurred: {str(e)}")
    
    # Schedule the next check
    Timer(60, check_for_new_messages).start()  # Check every 60 seconds

# Start monitoring the database for new messages
if __name__ == "__main__":
    initialize_last_message_time()
    check_for_new_messages()