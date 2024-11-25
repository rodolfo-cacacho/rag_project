import time
import os
import fnmatch
import telebot
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Timer
from datetime import datetime, timedelta

# Initialize the bot with your token
bot_token = "7498948634:AAEPlbjmMnM6U4V_GraoM8bak23QLCBSZeg"
bot = telebot.TeleBot(bot_token)

# Define the chat ID
chat_id = "7289085403"

# Path to monitor
CONVERSATION_DIR = "/home/ubuntu/docs/rag_clean_v1/bot/conversations"

# Initialize a list to keep track of files and their last modified times
files = []

# Loop through all the files in the directory to initialize the list
for filename in os.listdir(CONVERSATION_DIR):
    filepath = os.path.join(CONVERSATION_DIR, filename)

    # Check if it's a file (and not a directory)
    if os.path.isfile(filepath):
        if fnmatch.fnmatch(filepath, '*_conversation_history.txt'):
            # Get the last modified time
            last_modified_time = os.path.getmtime(filepath)
            
            # Convert the timestamp to a datetime object
            last_modified_datetime = datetime.fromtimestamp(last_modified_time)
            
            # Add the file and its last modified time to the list
            file_dict = {'file': filename, 'last_modified': last_modified_datetime}
            files.append(file_dict)
            
            print(f"File: {filename}, Last Modified: {last_modified_datetime}")

# Define the event handler for file modifications
class ConversationFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global files

        if event.is_directory:
            return
        
        if fnmatch.fnmatch(event.src_path, '*_conversation_history.txt'):
            filename = os.path.basename(event.src_path)
            filepath = event.src_path

            # Get the current time
            current_time = datetime.now()
            
            # Check if the file is already in the list
            file_entry = next((item for item in files if item['file'] == filename), None)
            if file_entry:
                # Calculate the time difference
                time_difference = current_time - file_entry['last_modified']
                
                # If the difference is more than 10 minutes, send a message
                if time_difference > timedelta(minutes=10):
                    user_info = filename.split('_')
                    username = user_info[0]
                    user_id = user_info[1]
                    message = f"User {username} ({user_id}) is using RAG Server"
                    bot.send_message(chat_id, message)

                # Update the last modified time in the list
                file_entry['last_modified'] = current_time
            else:
                # If the file is not in the list, add it and send a message
                file_dict = {'file': filename, 'last_modified': current_time}
                files.append(file_dict)

                # Send a message for the new file
                user_info = filename.split('_')
                username = user_info[0]
                user_id = user_info[1]
                message = f"User {username} ({user_id}) is using RAG Server"
                bot.send_message(chat_id, message)

# Monitoring setup for the conversation files
def start_monitoring():
    event_handler = ConversationFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=CONVERSATION_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Main execution
if __name__ == "__main__":
    start_monitoring()