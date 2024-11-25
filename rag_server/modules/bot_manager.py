import time
import logging
import traceback
from telebot.apihelper import ApiException
import sys

# Set up logging

ACTIVE_USER = None

logging.basicConfig(filename='bot_error.log', level=logging.ERROR, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ADMIN_CHAT_ID = "7289085403"  # Replace with your admin's chat ID

def notify_user(bot, user_id, message):
    try:
        bot.send_message(chat_id=user_id, text=message)
    except ApiException as e:
        print(f"Failed to send message to user {user_id}: {str(e)}")

def run_bot(bot):
    global ACTIVE_USER
    try:
        value_try = 1
        bot.polling(none_stop=True, interval=0, timeout=20)
    except ApiException as e:
        error_msg = f"Telegram API error: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        print(error_msg)
        sys.exit(1)  # Exit the script with a non-zero status to indicate failure
    except Exception as e:
        print('THIS IS WORKING - CRASHED')
        error_msg = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
        # logging.error(error_msg)
        # print(error_msg)
        print(f'Active User: {ACTIVE_USER} value: {value_try}')
        if ACTIVE_USER:
            notify_user(bot, ACTIVE_USER, "Das RAG-System ist abgestürzt und wird neu gestartet.")
        
        # Exit the script with a non-zero status to indicate failure
        sys.exit(1)
        
    return True

def manage_bot(bot):
    while True:
        if run_bot(bot):
            break
        print("Bot crashed. Restarting in 5 seconds...")
        notify_user(bot, "Bot is restarting...")
        time.sleep(5)

    # Explicitly exit with non-zero status if the loop exits unexpectedly
    sys.exit(1)