import os
import telebot
from dotenv import load_dotenv

load_dotenv()

API_KEY_TG = os.getenv('API_KEY_TG')
print(f'API Telegram: ',{API_KEY_TG})

bot = telebot.TeleBot(API_KEY_TG)

message = 'Die Probleme wurden behoben, und Sie können das System wieder nutzen. Vielen Dank für das Testen!'
user_id = '7415190825'

bot.send_message(chat_id=user_id, text=message)

user_id = '7289085403'

bot.send_message(chat_id=user_id, text=message)