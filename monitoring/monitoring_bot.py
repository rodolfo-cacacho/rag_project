import fnmatch
import telebot
from telebot import types
import prettytable as pt
import os
import time
import csv
import sys
from datetime import datetime
import re
import subprocess
from dotenv import load_dotenv
from telebot.util import smart_split
from database_manager import MySQLDB

# MySQL connection details
CONFIG_SQL_DB = {
    'user': 'root',
    'password': 'admin123',
    'host': 'localhost'
}

DB_NAME = 'data_rag'

sql_db_connector = MySQLDB(CONFIG_SQL_DB,DB_NAME)

load_dotenv()
# Initialize the bot with your token
API_KEY_TG_NOTIFY = os.getenv('API_KEY_TG_NOTIFY')

bot = telebot.TeleBot(API_KEY_TG_NOTIFY)

# Define the chat ID
chat_id = os.getenv('ADMIN_TG_ID')

# Path to the PID file and log file
PIDFILE = "/home/ubuntu/docs/logs/rag_process.pid"
LOGFILE = "/home/ubuntu/docs/logs/logfile.log"
USER_CSV = "/home/ubuntu/docs/rag_clean_v1/bot/conversations/users_db.csv"

def send_message(text):
    chunks = smart_split(text)
    for i in chunks:
        bot.send_message(chat_id, i)

@bot.message_handler(commands=['users'])
def send_users(message):
    try:
        # Query the user_db table from MySQL
        query = "SELECT username, name, DATE(date_added) as date_added, allowed FROM user_db"
        sql_db_connector.cursor.execute(f"USE {sql_db_connector.database_name}")
        sql_db_connector.cursor.execute(query)

        # Fetch all rows from the result
        users = sql_db_connector.cursor.fetchall()

        # Create a PrettyTable object and specify the columns
        table = pt.PrettyTable()
        table.field_names = ["username", "name", "date_added", "allowed"]

        # Add each row to the table
        for user in users:
            table.add_row([user[0], user[1], user[2], 'Yes' if user[3] else 'No'])

        # Convert table to string and send it as a message
        table_str = f"\n{table.get_string()}\n"
        send_message(f"Current Users:\n{table_str}")
    
    except Exception as e:
        send_message(f"An error occurred: {str(e)}")

@bot.message_handler(commands=['uptime'])
def send_uptime(message):
    try:
        # Run the `ps aux | grep python3` command to get the process info
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
        processes = result.stdout.decode().splitlines()

        # Filter the processes to find the relevant one
        for process in processes:
            if 'python3' in process and 'main_rag.py' in process:
                parts = process.split()
                pid = parts[1]  # PID is usually the second element
                start_time = parts[8]  # Adjust index if necessary

                current_year = time.strftime("%Y")

                if re.match(r'^\d{2}:\d{2}$', start_time):
                    # Case 1: Process started today (HH:MM)
                    process_start_time = time.mktime(time.strptime(
                        current_year + " " + time.strftime("%b %d") + " " + start_time, 
                        "%Y %b %d %H:%M"
                    ))

                elif re.match(r'^\w{3}\d{2}$', start_time):
                    # Case 2: Process started earlier this year (MMMDD)
                    process_start_time = time.mktime(time.strptime(
                        current_year + " " + start_time, 
                        "%Y %b%d"
                    ))

                elif re.match(r'^\d{4}$', start_time):
                    # Case 3: Process started in a different year (YYYY)
                    process_start_time = time.mktime(time.strptime(
                        start_time, 
                        "%Y"
                    ))

                # Calculate uptime
                uptime_seconds = time.time() - process_start_time
                uptime = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))

                send_message(f"RAG System Uptime: {uptime} (PID: {pid})")
                return

        # If no relevant process was found
        send_message("RAG System is not running.")

    except Exception as e:
        send_message(f"Error checking uptime: {str(e)}")

@bot.message_handler(commands=['prompts'])
def send_prompts(message):
    # Check if the message includes "total" to fetch all prompts
    fetch_all = "total" in message.text.lower()
    
    try:
        if fetch_all:
            # Fetch all prompts from the database, including the duration calculated from end_date and begin_date
            query = """
                SELECT u.name, u.username, p.question, p.answer, p.evaluation, p.comment, p.query_date, 
                       TIMESTAMPDIFF(SECOND, p.begin_date, p.end_date) as duration, p.total_tokens, p.context_used
                FROM prompts p 
                JOIN user_db u ON p.chat_id = u.id
            """
        else:
            # Fetch only today's prompts from the database
            today = datetime.now().date()
            query = f"""
                SELECT u.name, u.username, p.question, p.answer, p.evaluation, p.comment, p.query_date, 
                       TIMESTAMPDIFF(SECOND, p.begin_date, p.end_date) as duration, p.total_tokens, p.context_used
                FROM prompts p 
                JOIN user_db u ON p.chat_id = u.id 
                WHERE DATE(p.query_date) = '{today}'
            """
        
        # Execute the query
        sql_db_connector.cursor.execute(f"USE {sql_db_connector.database_name}")
        sql_db_connector.cursor.execute(query)

        # Fetch all rows from the result
        prompts = sql_db_connector.cursor.fetchall()

        if not prompts:
            send_message("No prompts found for today." if not fetch_all else "No prompts found.")
            return

        # Create the CSV file
        csv_filename = f"prompts_{'all' if fetch_all else 'today'}_{datetime.now().strftime('%Y%m%d')}.csv"
        csv_filepath = os.path.join("/home/ubuntu/docs/rag_clean_v1/bot/conversations", csv_filename)
        
        with open(csv_filepath, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Name", "Username", "Question", "Answer", "Evaluation", "Comment", "Query Date", "Duration (seconds)", "Total Tokens", "Context Used"])
            writer.writerows(prompts)
        
        # Send the CSV file to the user
        with open(csv_filepath, 'rb') as file:
            bot.send_document(message.chat.id, file)

        # Send a message with the count of prompts
        send_message(f"{len(prompts)} prompts have been sent.")

        # Delete the CSV file after it has been sent
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
    
    except Exception as e:
        send_message(f"An error occurred: {str(e)}")

@bot.message_handler(commands=['last_interaction'])
def send_last_interaction(message):
    # Extract the command parts
    command_parts = message.text.split()

    # Default values
    num_messages = 2
    person_name = None

    # Process the command to find name and number of messages
    for part in command_parts[1:]:
        if part.isdigit():
            num_messages = int(part)
        else:
            person_name = part.lower()  # Handle case insensitivity

    try:
        if person_name:
            # If person_name is provided, fetch messages for that user
            query = f"""
                SELECT u.name, m.chat_id, m.role, m.message, m.date
                FROM messages m
                JOIN user_db u ON m.chat_id = u.id
                WHERE LOWER(u.name) = %s
                ORDER BY m.date DESC
                LIMIT %s
            """
            sql_db_connector.cursor.execute(f"USE {sql_db_connector.database_name}")
            sql_db_connector.cursor.execute(query, (person_name, num_messages))
        else:
            # If no person name is provided, fetch the most recent messages across all users
            query = f"""
                SELECT u.name, m.chat_id, m.role, m.message, m.date
                FROM messages m
                JOIN user_db u ON m.chat_id = u.id
                ORDER BY m.date DESC
                LIMIT %s
            """
            sql_db_connector.cursor.execute(f"USE {sql_db_connector.database_name}")
            sql_db_connector.cursor.execute(query, (num_messages,))

        # Fetch the result from the query
        messages = sql_db_connector.cursor.fetchall()

        if not messages:
            send_message("No messages found.")
            return

        # Send messages as a formatted response
        for message_row in reversed(messages):
            user_name, chat_id, role ,content, date = message_row
            text_message = f"{user_name}_{chat_id} - {date}\n{role}: {content}\n"
            send_message(text_message)

    except Exception as e:
        send_message(f"An error occurred: {str(e)}")

# Store the state of the restart confirmation
restart_confirmation = {}

@bot.message_handler(commands=['restart_server'])
def ask_restart_confirmation(message):
    chat_id = message.chat.id
    
    # Send a confirmation message with inline keyboard buttons
    markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    markup.add('Yes', 'No')
    
    msg = bot.send_message(chat_id, "Are you sure you want to restart the server? (Yes/No)", reply_markup=markup)
    
    # Register the chat ID for confirmation state
    restart_confirmation[chat_id] = True
    
    # Listen for the response
    bot.register_next_step_handler(msg, process_restart_confirmation)

def process_restart_confirmation(message):
    chat_id = message.chat.id
    
    if chat_id not in restart_confirmation or not restart_confirmation[chat_id]:
        # If the chat ID is not in the confirmation state, ignore the message
        return
    
    # Reset the confirmation state
    restart_confirmation[chat_id] = False
    
    response = message.text.strip().lower()
    
    # Remove the keyboard markup
    markup_remove = types.ReplyKeyboardRemove()
    
    if response in ['yes', 'y']:
        bot.send_message(chat_id, "Restarting the server...", reply_markup=markup_remove)
        # Restart the server using sudo reboot command
        subprocess.call(['sudo', 'reboot'])
    elif response in ['no', 'n']:
        bot.send_message(chat_id, "Server restart canceled.", reply_markup=markup_remove)
    else:
        bot.send_message(chat_id, "Invalid response. Please send /restart_server again if you want to restart.", reply_markup=markup_remove)


# Store the selected user and date filters
chat_filters = {}

@bot.message_handler(commands=['chat'])
def select_user(message):
    try:
        # Fetch all users from user_db
        query = "SELECT id, name, username FROM user_db"
        sql_db_connector.cursor.execute(f"USE {sql_db_connector.database_name}")
        sql_db_connector.cursor.execute(query)

        users = sql_db_connector.cursor.fetchall()

        if not users:
            send_message("No users found.")
            return

        # Create a markup keyboard with the available users (name + username + id)
        markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
        for user in users:
            user_display = f"{user[1]} ({user[2]}, ID: {user[0]})"  # Format: Name (username, ID: id)
            markup.add(user_display)

        # Store the chat ID to track which user is selected
        chat_filters[message.chat.id] = {"user_id": None, "username": None, "date_filter": None}

        # Ask the user to select a user from the list
        msg = bot.send_message(message.chat.id, "Select a user:", reply_markup=markup)
        bot.register_next_step_handler(msg, select_date_filter)

    except Exception as e:
        send_message(f"An error occurred: {str(e)}")


def select_date_filter(message):
    try:
        # Extract user_id from the message text (assuming format "Name (username, ID: id)")
        selected_text = message.text
        user_id = int(selected_text.split('ID: ')[1].strip(')'))  # Extract ID from the message

        # Store the selected user_id and username in chat_filters
        chat_filters[message.chat.id]["user_id"] = user_id
        chat_filters[message.chat.id]["username"] = selected_text.split('(')[1].split(',')[0]

        # Create a markup keyboard for date filtering (today or total)
        markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
        markup.add('Today', 'Total')

        # Ask the user to select a date filter
        msg = bot.send_message(message.chat.id, "Do you want to filter for today's messages or all messages?", reply_markup=markup)
        bot.register_next_step_handler(msg, fetch_and_send_messages)

    except Exception as e:
        send_message(f"An error occurred: {str(e)}")

def fetch_and_send_messages(message):
    # Save the selected date filter in chat_filters
    chat_filters[message.chat.id]["date_filter"] = message.text.lower()

    try:
        # Fetch user details
        user_id = chat_filters[message.chat.id]["user_id"]
        date_filter = chat_filters[message.chat.id]["date_filter"]

        # Build the query based on the date filter
        if date_filter == "today":
            today = datetime.now().date()
            query = f"""
                SELECT u.name, u.username, m.role, m.message, m.date, m.reply_message_id, m.prompt_id
                FROM messages m 
                JOIN user_db u ON m.chat_id = u.id 
                WHERE u.id = %s AND DATE(m.date) = '{today}'
            """
        else:  # "total"
            query = """
                SELECT u.name, u.username, m.role, m.message, m.date, m.reply_message_id, m.prompt_id
                FROM messages m 
                JOIN user_db u ON m.chat_id = u.id 
                WHERE u.id = %s
            """

        # Execute the query using user_id
        sql_db_connector.cursor.execute(query, (user_id,))
        messages = sql_db_connector.cursor.fetchall()  # Fetch all rows to clear the result

        if not messages:
            send_message(f"No messages found for user ID {user_id} on the selected date.")
            return

        # Create the CSV file
        csv_filename = f"messages_{user_id}_{'today' if date_filter == 'today' else 'total'}_{datetime.now().strftime('%Y%m%d')}.csv"
        csv_filepath = os.path.join("/home/ubuntu/docs/rag_clean_v1/bot/conversations", csv_filename)

        with open(csv_filepath, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Name", "Username", "Role", "Message", "Date", "Reply Message ID", "Prompt ID"])
            writer.writerows(messages)

        # Send the CSV file to the user
        with open(csv_filepath, 'rb') as file:
            bot.send_document(message.chat.id, file)

        # Send a message with the count of messages
        send_message(f"{len(messages)} messages have been sent.")

        # Delete the CSV file after it has been sent
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)

    except Exception as e:
        send_message(f"An error occurred: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Send initial notification
    if len(sys.argv) > 1:
        message = sys.argv[1]
        send_message(message)
    else:
        send_message("RAG System Notification: No specific message provided")

    # Start the bot polling for interaction
    bot.polling()
