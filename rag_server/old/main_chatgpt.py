import os
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import openai
from chat import Chat,ChatManager
from rag_retrieval import retrieve_context,get_dates_documents,build_context
from user_data_base import add_user_to_csv,is_user_allowed,get_allowed_users
from dotenv import load_dotenv
import pandas as pd
from database_manager import MySQLDB,ChromaDBConnector
from PIL import Image                                                                                
import base64
import unicodedata


load_dotenv()

#API_KEY = os.getenv('API_KEY')
API_KEY_CGPT = os.getenv('API_KEY_CGPT')
API_KEY_TG = os.getenv('API_KEY_TG')
API_KEY_TG_NOTIFY = os.getenv('API_KEY_TG_NOTIFY')
AUTH_TELEBOT_PWD = os.getenv('AUTH_TELEBOT_PWD')


print(f'API Telegram: ',{API_KEY_TG})
bot = telebot.TeleBot(API_KEY_TG)

print(f'API ChatGPT: ',{API_KEY_CGPT})

openai.api_key = API_KEY_CGPT

USER_DB_FILE = 'bot/conversations/users_db.csv'
BOT_NAME = 'THWS Bau Bot'
DIST_THRESHOLD = 500

user_send_pwd = []

authorized_users = get_allowed_users(USER_DB_FILE)
print(authorized_users)

print(f'cwd: {os.getcwd()}')

DATE_DB_FILE = 'data/documents/metadata/Files_date_version.csv'
FILE_CHAT_GPT_PROMPTS = f'bot/conversations/chatgpt_prompts.csv'
LOCATION_CHAT_CONV = 'bot/conversations'

date_database = pd.read_csv(DATE_DB_FILE)

available_documents = date_database['type'].unique()

print(f'Documents {available_documents}')

# print(date_database.head(20))

conversation_history = [
    {"role": "system",
     "content": """You are a helpful Retrieval-Augmented Generation (RAG) system specializing in answering questions related to government financing of efficient buildings in Germany. Your primary function is to clarify questions about the set of regulations and funding programs. Follow these guidelines to provide the best possible assistance:

	1.	Scope and Relevance:
	•	Focus exclusively on construction-related topics, specifically government financing and regulations for efficient buildings in Germany.
	•	If a query is unrelated, politely inform the user that you specialize in construction-related topics in Germany.
	2.	Language:
	•	Prefer responses in English and German.
	•	Use the informal ‘you’ (du) unless the user explicitly requests a formal tone.
	3.	Response Style:
	•	Provide precise and concise answers.
	•	Maintain a friendly and approachable tone.
	•	Include casual greetings and friendly messages to make interactions more human-like.
	4.	Sources and Credibility:
	•	Whenever possible, include the source of the information in your responses.
	•	Ensure the information is accurate and up-to-date, especially when referring to official documents and funding schemes.
	5.	Handling Unrelated Queries:
	•	If asked about unrelated topics, respond with:
	•	“I am a specialized system for answering questions about government financing and regulations for efficient buildings in Germany. How can I assist you with that topic?”"""}
]

chroma_db_connector = ChromaDBConnector(
        storage_path="data/storage/embeddings",
        collection_name="rag_faq",
        embedding_model = 'german'
)

# MySQL connection details
CONFIG_SQL_DB = {
    'user': 'root',
    'password': 'admin123',
    'host': 'localhost'
}

DB_NAME = 'data_rag'
TABLE_NAME = 'embedding_table'
    

sql_db_connector = MySQLDB(CONFIG_SQL_DB,DB_NAME)

MX_RESULTS = 5

user_conversations = {}
chat_manager = ChatManager(FILE_CHAT_GPT_PROMPTS,LOCATION_CHAT_CONV)

# Define a global variable to track the active user
ACTIVE_USER = None

def track_active_user(func):
    def wrapper(message):
        global ACTIVE_USER
        try:
            # Update ACTIVE_USER with the user ID from the message
            ACTIVE_USER = message.from_user.id
            print(f"ACTIVE_USER set to: {ACTIVE_USER}")
            
            # Call the actual function
            result = func(message)
            
            # If everything works, set ACTIVE_USER to None
            ACTIVE_USER = None
            print(f"ACTIVE_USER reset to: {ACTIVE_USER}")
            
            return result
        except Exception as e:
            # Log the error or handle it as needed
            print(f"An error occurred: {e}")
            # Optionally, re-raise the exception if you want the bot to crash
            raise
    return wrapper

# Define the cleaning function
def clean_text(text):
    # Normalize the text to combine characters like 'a' + '\u0308' into 'ä'
    normalized_text = unicodedata.normalize('NFC', text)
    return normalized_text.strip()

def simple_llm_query(text,system_instruction = 'You are a helpful assistant.'):
    
    q_conversation = [
        {"role":"system",
         "content":system_instruction}
    ]
    q_chat = {"role": "user", "content": text}
    q_conversation.append(q_chat)

    response = openai.chat.completions.create(
        model="gpt-4o",  # Change this to the desired model (e.g., "davinci" or "curie")
        messages=q_conversation,
        max_tokens=50  # Adjust the maximum number of tokens for the response
    )
    answer = response.choices[0].message.content

    return answer

def update_date_conversation(conversation,query='Text'):

    system_instructions = '''Based on a given text, infer the date which is the question talking about. If no date is inferred, respond: "No date found", if yes, say the date in the following format: dd/mm/yyyy. Just answer with the date or the text. When no exact day can inferred, assume last day of the given month. If just the year is mention, say the last day of the year.
    If a date in the future is mentioned, return that no date was found. '''

    query_send = f'''The text is the following:
    {query}'''

    retrieved_date = simple_llm_query(text=query_send,system_instruction=system_instructions)
    ## Clean date
    retrieved_date = retrieved_date.strip()
    

    if retrieved_date.replace('.','').lower() != 'no date found':
        # print(f'Date was updated with: {retrieved_date}')
        conversation.update_project_date(retrieved_date)
    # else:
    #     print(retrieved_date)
    
    return conversation.project_date

def send_long_message(bot, message, answer, max_length=4000):
    # Split the answer into chunks with a maximum length of max_length
    chunks = [answer[i:i + max_length] for i in range(0, len(answer), max_length)]

    messages_sent = []
    # Loop through each chunk and send it as a reply
    for chunk in chunks:
        message_sent = bot.reply_to(message, chunk)
        messages_sent.append(message_sent)
    
    return messages_sent


# Define a function to check if a user is authorized
def is_user_authorized(username,user_id):
    if username is None:
        return str(user_id) in authorized_users
    else:
        return username in authorized_users

def add_question(conversation,question):
    chat = {"role": "user", "content": question}

    conversation.append(chat)
    return conversation

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def add_question_context(conversation,question,context):

    if context == None:
        prompt = "No context was found for the question."
        chat = {"role": "user", "content":prompt}
        ask = False
    else:
        ask = True
        prompt = f"""
        Answer the following question using the context below. If the quesion cannot be answered with the context, say you don't know, if unrelated, mention your purpose/goal.
        Always answer in the language that the questions was asked. Mention which source was used to answer the question.
        
        Question: {question}

        Context:\n\n"""

        content = []
        dict_images = []
        n_images = 0
        for i in context:
            context_i = f"{i['source']} Pages:{i['pages']}\n{i['text']}"
            prompt += context_i + '\n========== ========== ========== ==========\n'
            tables = i['paths']

            for index,tab in enumerate(tables):
                n_images+=1
                clean_tab_name = clean_text(tab)
                encoded_img = encode_image(clean_tab_name)
                dic_images = {'type':'image_url','image_url':{'url': f"data:image/png;base64,{encoded_img}"}}
                dict_images.append(dic_images)
                # print(f'Table {index}')
                # img = Image.open(tab)
                # img.show() 


        if n_images>0:
            prompt_text = {'type':'text','text':prompt}
            content.append(prompt_text)
            content.extend(dict_images)
            chat = {"role": "user", "content":content}
        else:
            chat = {"role": "user", "content":prompt}

    conversation.append(chat)

    # print(f'Prompt\n{prompt}')
    
    return conversation,ask


    # content = []
    # question_text = {'type':'text','text':question}

    # dict_images = []
    # for i in paths:
    #     encoded_img = encode_image(i)
    #     dic_images = {'type':'image_url','image_url':{'url': f"data:image/png;base64,{encoded_img}"}}
    #     dict_images.append(dic_images)

    # content.append(question_text)

    # if len(paths) > 0:
    #     content.extend(dict_images)

def update_conversation(conversation,response):

    if isinstance(conversation[-1]["content"], list):
        for item in conversation[-1]["content"]:
            if item["type"] == "text":
                conversation[-1]["content"] = item["text"]
                break

        # print('removed images from chatgpt history')
        # print(f'Conversation: {conversation}')


    chat = {"role":"assistant", "content":response}
    conversation.append(chat)
    return conversation

def gen_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton("Ja ✅", callback_data="cb_yes"),
                               InlineKeyboardButton("No ❌", callback_data="cb_no"))
    return markup

def yes_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Ja ✅",callback_data = 'cb_responded'))
    return markup

def no_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Nein ❌",callback_data = 'cb_responded'))
    return markup

# Define a function to ask a question to ChatGPT
def ask_question(conversation,question):
    try:
        conversation = add_question(conversation,question)
        response = openai.chat.completions.create(
            model="gpt-4o",  # Change this to the desired model (e.g., "davinci" or "curie")
            messages=conversation,
            max_tokens=2000  # Adjust the maximum number of tokens for the response
        )
        answer = response.choices[0].message.content
        conversation = update_conversation(conversation,answer)
        return conversation,answer
    
    except Exception as e:
        return str(e)


# Define a function to ask a question to ChatGPT
def ask_question_with_context(conversation,question,context):
    # try:
    conversation,ask = add_question_context(conversation,question,context)
    if ask:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Change this to the desired model (e.g., "davinci" or "curie")
            messages=conversation,
            max_tokens=2000  # Adjust the maximum number of tokens for the response
        )
        answer = response.choices[0].message.content
        # print(f'Answer was obtained: {answer}')
        conversation = update_conversation(conversation,answer)
        return conversation,answer
    else:
        answer = 'Es wurde kein Kontext zu den gestellten Fragen gefunden. Bitte überprüfe die Frage oder formuliere sie gegebenenfalls um.'
        conversation = update_conversation(conversation,answer)
        return conversation,answer
    
    # except Exception as e:
    #     return str(e)
    
# Define a handler for the /start command
@bot.message_handler(commands=['getaccess'])
@track_active_user
def get_access(message):
    global chat_manager
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    user_id = message.chat.id
    user_chat = chat_manager.get_chat(user=user_name,first_name=name_user,id = user_id)
    cid = message.chat.id
    mid = message.message_id
    user_chat.add_message(role = user_name,content = message.text,chat_id = cid, message_id = mid)
    exist_user,allowed_user = is_user_allowed(USER_DB_FILE,telegram_username=user_name,telegram_id=user_id)
    # print(f'exist {exist_user} allowed {allowed_user}')

    if allowed_user == 'False':
        message_bot = f'Willkommen {name_user} bei {BOT_NAME}! Du darfst nicht chatten, bitte gib das Passwort ein, um Zugriff auf den Bot zu erhalten.'
        user_send_pwd.append(user_id)
    elif exist_user == 'True' and allowed_user == 'True':
        message_bot = f"Willkommen {name_user} bei {BOT_NAME}! Du darfst bereits chatten."
    # print(f'exist? {exist_user}')
    if exist_user == 'False':
        # print('added user')
        add_user_to_csv(file_path=USER_DB_FILE,username=user_name,id=user_id,name=name_user,allowed=False)

    replied_msg = bot.reply_to(message, message_bot)
    user_chat.add_message(role = 'assistant',content = message_bot,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)

# Define a handler for the /start command
@bot.message_handler(commands=['start'])
@track_active_user
def send_welcome(message):
    global chat_manager
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    cid = message.chat.id
    mid = message.message_id
    user_chat = chat_manager.get_chat(user=user_name,first_name=name_user,id=cid)
    # print(f'Revising initialization:\n{user_chat}')
    user_chat.add_message(role = user_name,content = message.text,chat_id = cid, message_id = mid)
    message_bot = f"Willkommen {message.from_user.first_name} bei {BOT_NAME}! Schreib etwas, um mit dem Chatten zu beginnen."
    replied_msg = bot.reply_to(message, message_bot)
    user_chat.add_message(role = 'assistant',content = message_bot,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)

# Define a handler for the /end command
@bot.message_handler(commands=['end'])
@track_active_user
def send_end(message):
    global chat_manager
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    cid = message.chat.id
    mid = message.message_id
    user_chat = chat_manager.get_chat(user=user_name,first_name=name_user,id=cid)
    user_chat.add_message(role = user_name,content = message.text,chat_id = cid, message_id = mid)
    message_bot = f"Vielen Dank {message.from_user.first_name}, dass du den {BOT_NAME} benutzt hast! Es war mir eine Freude, dir zu helfen."
    replied_msg = bot.reply_to(message, message_bot)
    user_chat.add_message(role = 'assistant',content = message_bot,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)


# Define a handler for the /reset command
@bot.message_handler(commands=['reset'])
@track_active_user
def send_reset(message):
    global chat_manager
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    cid = message.chat.id
    mid = message.message_id
    user_chat = chat_manager.get_chat(user=user_name,first_name=name_user,id=cid)
    user_chat.add_message(role = user_name,content = message.text,chat_id = cid, message_id = mid)
    message_bot = f"Deine Gesprächshistorie wird zurückgesetzt!"
    replied_msg = bot.reply_to(message, message_bot)
    user_chat.add_message(role = 'assistant',content = message_bot,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)
    user_chat.reset_gpt_conversation_history()

# Define a handler for the /reset command
@bot.message_handler(commands=['cancel'])
@track_active_user
def send_cancel(message):
    global chat_manager
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    cid = message.chat.id
    mid = message.message_id
    user_chat = chat_manager.get_chat(user=user_name,first_name=name_user,id=cid)
    user_chat.add_message(role = user_name,content = message.text,chat_id = cid, message_id = mid)
    message_bot = f"Vorgang abgebrochen!"
    replied_msg = bot.reply_to(message, message_bot)
    user_chat.add_message(role = 'assistant',content = message_bot,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)
    if cid in user_send_pwd:
        user_send_pwd.remove(cid)

@bot.callback_query_handler(func=lambda call: True)
@track_active_user
def callback_query(call):
    global chat_manager
    cid = call.message.chat.id
    mid = call.message.message_id
    username = call.message.chat.username
    user_name = call.message.from_user.username
    name_user = call.message.from_user.first_name

    user_chat = chat_manager.get_chat(user_name,name_user,id=cid)
    text_answer = 'Deine Antwort war:'
    # print(f'Response eval: {call.data} - username: {user_name} name: {name_user} cid: {cid} mid: {mid}')
    if call.data == "cb_yes":
        bot.answer_callback_query(call.id, "Antwort ist Ja")
        bot.edit_message_text(chat_id=cid, message_id=mid,
                          text=text_answer, reply_markup=yes_markup())
        user_chat.edit_prompt_eval(cid_answer = str(cid),
                                   mid_answer = str(mid - 1),
                                   value = 'good',
                                   column = 'evaluation')
        # bot.delete_message(cid,mid)
        # bot.send_message(cid,"Your answer was:", reply_markup=yes_markup())
    elif call.data == "cb_no":
        bot.answer_callback_query(call.id, "Antwort ist Nein")
        bot.edit_message_text(chat_id=cid, message_id=mid,
                          text=text_answer, reply_markup=no_markup())
        user_chat.edit_prompt_eval(cid_answer = str(cid),
                                   mid_answer = str(mid - 1),
                                   value = 'bad',
                                   column = 'evaluation')
    elif call.data == 'cb_responded':
        bot.answer_callback_query(call.id,"Bereits beantwortet")
        # bot.delete_message(cid,mid)
        # bot.send_message(cid,"Your answer was:", reply_markup=no_markup())

@bot.message_handler(func=lambda message: message.reply_to_message)
@track_active_user
def reply_msg(message):
    global chat_manager
    cid = message.chat.id
    mid = message.reply_to_message.message_id
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    comment = message.text
    print(f'Dein Kommentar war: {comment}')
    user_chat = chat_manager.get_chat(user_name,name_user,id = cid)
    user_chat.edit_prompt_eval(cid_answer = str(cid),
                                              mid_answer = str(mid),
                                              value = comment,
                                              column = "comment")
    

# Define a handler for regular messages
@bot.message_handler(func=lambda message: True)
@track_active_user
def echo_all(message):
    global user_conversations 
    global chat_manager
    global authorized_users
    # Check if the user is authorized
    user_name = message.from_user.username
    name_user = message.from_user.first_name
    user_id = message.from_user.id
    message_rcvd = message.text
    chat_id = message.chat.id
    message_id = message.message_id
    # print(f'id: {user_id} username: {user_name} name: {name_user}')

    user_chat = chat_manager.get_chat(user_name,name_user,id=user_id)

    if user_id in user_send_pwd:
        if message_rcvd == AUTH_TELEBOT_PWD:
            message_to_send = f'Richtiges Passwort! Du kannst jetzt mit {BOT_NAME} chatten!'
            add_user_to_csv(file_path=USER_DB_FILE,username=user_name,id=user_id,name=name_user,allowed=True)
            authorized_users = get_allowed_users(USER_DB_FILE)
            user_send_pwd.remove(user_id)

        else:
            message_to_send = f'Falsches Passwort! Versuche es erneut! Sende /cancel, um die Anfrage zu beenden!'

        replied_msg = bot.reply_to(message, message_to_send)
        user_chat.add_message(role = "assistant",content = message_to_send,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)


    elif is_user_authorized(user_name,user_id):
            
        # Get the message text
        selected_keys = ['role', 'content']
        # Extract dictionaries containing the specified variables
        conversation_history = [{key: d[key] for key in selected_keys} for d in user_chat.gpt_history]

        # gpt_history = {key: user_chat.gpt_history[key] for key in selected_keys}
        input_text = message_rcvd
        user_chat.add_message(role = user_name,content = message_rcvd,chat_id = chat_id,message_id = message_id,gpt_history = True)


        # Check if the input text is too short
        if len(input_text.split()) < 2 or len(input_text) < 5:
            answer = "Deine Nachricht ist zu kurz, um verstanden zu werden. Bitte gib mehr Informationen an."
            replied_msg = bot.reply_to(message, answer)
            user_chat.add_message(role = "assistant",content = answer,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = True)
            return
        
        date_project = update_date_conversation(conversation=user_chat,query=input_text)

        dates_files = get_dates_documents(date_project,date_database)
        # print(f'DATES QUERY {dates_files}')

        results = retrieve_context(chroma_connector=chroma_db_connector,
                                   sql_connector=sql_db_connector,
                                   sql_table=TABLE_NAME,
                                   query=message_rcvd,
                                   max_results=MX_RESULTS,
                                   print_t=False,
                                   distance_threshold=DIST_THRESHOLD,
                                   retrieve_date_documents = dates_files)
        results = build_context(results,sql_db_connector,TABLE_NAME)
        # Generate a response using ChatGPT
        conversation_history,answer = ask_question_with_context(conversation_history,input_text,results)

        # Send the response back to the user
        # replied_msg = bot.reply_to(message, answer)
        # Usage
        replied_msgs = send_long_message(bot, message, answer)
        answer_mid = [i.id for i in replied_msgs]
        # answer_mid = replied_msg.id
        answer_cid = replied_msgs[0].chat.id
        quality_resp = bot.send_message(message.chat.id, "War das eine gute Antwort?", reply_markup=gen_markup())
        user_chat.add_message(role = "assistant",content = answer,chat_id = answer_cid,message_id = answer_mid,gpt_history = True)
        answer_mid.append(quality_resp.id)
        # print(user_chat.gpt_history)
        user_chat.save_prompts(question = input_text,
                               context = results,
                               answer = answer,
                               evaluation = "",
                               comment = "",
                               mid_question = message_id,
                               cid_question = chat_id,
                               mid_answer = answer_mid,
                               cid_answer = answer_cid,
                               name_user = name_user,
                               user_name = user_name)
        user_chat.reset_gpt_conversation_history(total = False)
        # print(f'GPT History size: {len(user_chat.gpt_history)}')

    else:
        answer = f'Tut mir leid, du bist nicht berechtigt, mit diesem Bot zu interagieren. Verwende den Befehl /getaccess, um eine Erlaubnis anzufordern.'
        replied_msg = bot.reply_to(message, answer)
        user_chat.add_message(role = "assistant",content = answer,chat_id = replied_msg.chat.id,message_id = replied_msg.id,gpt_history = False)


# Start the bot
bot.polling()