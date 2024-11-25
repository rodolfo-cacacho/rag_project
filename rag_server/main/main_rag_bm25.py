import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)


import telebot
from telebot.util import smart_split
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import openai
from rag_server.modules.rag_retrieval_bm25 import retrieve_context,get_dates_documents,build_context,get_data_ids
from rag_server.modules.user_manager import add_user_to_db,is_user_allowed,get_allowed_users
from dotenv import load_dotenv
import pandas as pd
from utils.MySQLDB_manager import MySQLDB
from utils.pinecone_hybrid_connector import PineconeDBConnectorHybrid
from utils.embedding_handler import EmbeddingHandler
import base64
import unicodedata
import sys
import time
import logging
import traceback
from telebot.apihelper import ApiException
import threading
import json
from rag_server.modules.chat import ChatManager
from functools import wraps
from datetime import datetime
from textwrap import dedent
from pydantic import BaseModel
import re
import getpass
from config import (DB_NAME,CONFIG_SQL_DB,METADATA_FILE_PATH,BOT_NAME,
                    SQL_CHUNK_TABLE,SQL_VOCAB_BM25_TABLE,
                    MAX_TOKENS,SUFFIX,INDEX_NAME,
                    EMBEDDING_MODEL,EMBEDDING_MODEL_API,EMBEDDING_MODEL_DIM,EMBEDDING_MODEL_RET_TASK,
                    ALPHA_VALUE,GEN_PROMPTS,
                    SQL_USER_TABLE_SCHEMA,SQL_USER_TABLE,
                    SQL_MESSAGES_TABLE,SQL_MESSAGES_TABLE_SCHEMA,
                    SQL_PROMPTS_TABLE,SQL_PROMPTS_TABLE_SCHEMA,
                    MX_RESULTS,DIST_THRESHOLD,MX_RESULTS_QUERY,USERS_SERVERS)

load_dotenv()


API_KEY_CGPT = os.getenv('API_KEY_CGPT')
if getpass.getuser() in USERS_SERVERS:
    API_KEY_TG = os.getenv('API_KEY_TG')
else:
    API_KEY_TG = os.getenv('API_KEY_TG_BBOT')
del getpass
AUTH_TELEBOT_PWD = os.getenv('AUTH_TELEBOT_PWD')
API_PINE_CONE = os.getenv('API_PINE_CONE')


print(f'API Telegram: ',{API_KEY_TG})
bot = telebot.TeleBot(API_KEY_TG)

print(f'API ChatGPT: ',{API_KEY_CGPT})
print(f'API Pinecone: ',{API_PINE_CONE})

openai.api_key = API_KEY_CGPT


user_send_pwd = []

print(f'cwd: {os.getcwd()}')

date_database = pd.read_csv(METADATA_FILE_PATH)

MIN_DATE_DB = pd.to_datetime(date_database['date'], format="%d/%m/%Y", errors='coerce', dayfirst=True).min().strftime('%d/%m/%Y')

available_documents = date_database['type'].unique()

print(f'Documents {available_documents}')

#########################################################################
#
#       DATABASE LOGIC INITIALIZATION
#
#########################################################################

pine_cone_connector = PineconeDBConnectorHybrid(api_key=API_PINE_CONE,
                                          index_name=INDEX_NAME,
                                          dimension= EMBEDDING_MODEL_DIM)

sql_db_connector = MySQLDB(CONFIG_SQL_DB,DB_NAME)

# Create the Users table if it doesn't exist
sql_db_connector.create_table(SQL_USER_TABLE, SQL_USER_TABLE_SCHEMA)
sql_db_connector.create_table(SQL_PROMPTS_TABLE,SQL_PROMPTS_TABLE_SCHEMA )
sql_db_connector.create_table(SQL_MESSAGES_TABLE,SQL_MESSAGES_TABLE_SCHEMA)

authorized_users = get_allowed_users(sql_db_connector)

print('Authorized Users: ')
for i in authorized_users:
    print(f'{i}')


chat_manager = ChatManager(db_connector = sql_db_connector)

# Define a global variable to track the active user
ACTIVE_USER = None

embed_handler = EmbeddingHandler(model_name=EMBEDDING_MODEL,
                                 use_api=EMBEDDING_MODEL_API)


#########################################################################
#
#       FUNCTION DECLARATION
#
#########################################################################

def clean_and_normalize_response(raw_response):
    """
    Cleans and normalizes a JSON API response by decoding Unicode and normalizing text only if necessary.

    Parameters:
    - raw_response: str, the raw JSON response as a string with escaped Unicode characters.

    Returns:
    - normalized_data: List of dictionaries, the normalized JSON data.
    """
    # Step 1: Check if the response contains Unicode escape sequences
    if not re.search(r'\\u[0-9a-fA-F]{4}', raw_response):
        # If no Unicode sequences are found, assume no processing is needed
        return raw_response

    # Step 2: Decode Unicode escape sequences
    decoded_response = raw_response.encode('utf-8').decode('unicode_escape')
        
    return decoded_response

class Context(BaseModel):
    source: str
    pages: str
    ids: list[str]
    text: str

class Rag_reponse(BaseModel):
    context_used: list[Context]
    answer: str

class user_query_format(BaseModel):
    prompts : list[str]
    key_terms: list[str]
    date_retrieved : str

class user_query_format2(BaseModel):
    improved_query : str
    intent : str
    key_terms: list[str]
    alernate_queries: list[str]
    date_retrieved : str

class user_query_format_alpha(BaseModel):
    alpha : float
    synonyms: list[str]

def parse_rag_response_to_dict(answer: str) -> dict:
    """
    Parse the structured output of RAG into a dictionary format.

    :param answer: The raw JSON-like string from the RAG response.
    :return: A dictionary containing the 'answer' and 'context_used' fields.
    """
    try:
        # Convert the string to a dictionary (assuming it's a valid JSON-like string)
        parsed_output = json.loads(answer)

        # Extract the context_used, if it exists, else return an empty list
        context_used = parsed_output.get('context_used', [])

        # Ensure context_used is a list of dictionaries (if it's empty, it will stay as an empty list)
        if not context_used:
            context_used = []
        
        # Create the final dictionary with answer and context_used
        response_dict = {
            "answer": parsed_output.get('answer', ''),
            "context_used": context_used
        }

        return response_dict

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing the response: {e}")
        # Return an empty dictionary if parsing fails
        return {"answer": "", "context_used": []}


def display_date_to_user(date_obj):
    # Convert to user-friendly string format when displaying to the user
    formatted_date = date_obj.strftime('%d/%m/%Y')
    return formatted_date

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
        print(error_msg)
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

##################################################################
#
#    WRAPPERS
#
##################################################################

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

def store_user_message(func):
    @wraps(func)
    def wrapper(message, *args, **kwargs):
        global chat_manager

        # Extract user info from the message object
        user_name = message.from_user.username
        name_user = message.from_user.first_name
        user_id = message.chat.id
        date_message = message.date
        mid = message.message_id

        # Get or create the user's chat object
        # get_chat(self, user, first_name, user_id)
        user_chat = chat_manager.get_chat(user=user_name, first_name=name_user, user_id=user_id)

        # Store the user's message in the chat
        user_chat.add_message(role='user', content=message.text, message_id=mid)

        # Pass the user_chat object to the original handler function
        return func(message, user_chat, *args, **kwargs)

    return wrapper

# Define the cleaning function
def clean_text(text):
    # Normalize the text to combine characters like 'a' + '\u0308' into 'ä'
    normalized_text = unicodedata.normalize('NFC', text)
    return normalized_text.strip()

def send_typing_action(bot, chat_id, stop_event):
    """
    Function to keep sending the 'typing' action every 4 seconds.
    Stops when the main action is completed.
    """
    while not stop_event.is_set():
        bot.send_chat_action(chat_id=chat_id, action='typing')
        time.sleep(4) 

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

def preprocess_user_query(user_chat,query,alt_queries = GEN_PROMPTS):

    # system_instructions = f"""You are a specialized system that analyzes user queries and helps extract important information of the query. Do the following 3 tasks: Generate alterantive user prompt, extract key terms and synonyms of user's query and extract date of the query that user is referring to.
    # 1.	Intent Recognition:
    # - Identify if the user is asking about requirements, funding, compliance, or technical details.
    # - Focus on specific terms related to technology (e.g., Wärmepumpen, U-Wert).
	# 2.	Key Terms:
    # - Extract key terms related to the subject (e.g., Wärmepumpen, Förderungen, U-Wert).
	# 3.	Generate Alternate Queries:
    # - Create {alt_queries} alternate queries using:
	#     1.	Synonyms or similar terms.
	#     2.	Different sentence structures.
	#     - Ensure they maintain the user’s intent and ensure the generated prompts are in ** german **.
    # 4. Extract key terms mentioned in the prompt, in which the user is interested of knowing about. Give a list of the terms and some synonyms.
    # 5. Based on the user's prompt, infer the date which is the question talking about. If no date is inferred, respond: "No date found", if yes, say the date in the following format: dd/mm/yyyy. Just answer with the date or the text. When no exact day can inferred, assume last day of the given month. If just the year is mention, say the last day of the year.
    # If a date in the future is mentioned, return that no date was found. 
	# 6.	Output:
    # - List of alternative generated prompts.
    # - List of key terms and synonyms.
    # - Extracted date of the user's prompt."""

    system_instructions = f"""You are a system designed to enhance user queries for an information retrieval system using BM25 and Semantic Search. Perform the following tasks on the given query:

1. **Error Checking and Rephrasing**:
   - Review the user's query for any grammatical, spelling, or contextual errors.
   - Rephrase the query for better clarity and accuracy while maintaining its original intent.

2. **Intent Recognition**:
   - Determine the user's intent (e.g., requirements, funding, compliance, technical details).
   - Highlight terms or topics that suggest the purpose of the query (e.g., "Wärmepumpen", "Förderungen", "U-Wert").

3. **Key Terms and Synonyms**:
   - Extract key terms directly relevant to the query.
   - For each key term, provide a list of synonyms or related terms to improve search coverage.

4. **Query Expansion**:
   - Generate **{alt_queries}** alternate queries.
   - Use the rephrased query as the base for expansion.
   - Avoid using new or big compound words when possible.
   - Incorporate synonyms, related terms, and varied sentence structures while maintaining the query's original intent.
   - Ensure all generated queries are in **German**.

5. **Date Extraction**:
   - Infer any dates mentioned in the user's query.
   - If no specific day is provided, assume the last day of the mentioned month or year.
   - If the query refers to a future date, respond with "No date found."
   - Output dates in the format: dd/mm/yyyy.
   
   Output:
   1. Rephrased query
   2. Intent of query
   3. List of Key terms
   4. List of alternate queries
   5. Extracted date: "No date found" or dd/mm/yyyy"""

    prompt = f"Analyze the user's question: {query}"

    answer = call_gpt_api_with_single_prompt(instructions=dedent(system_instructions),
                                             prompt= dedent(prompt),
                                             response_format= user_query_format2
                                             )

    answer = json.loads(answer)

    rephrased_query = answer['improved_query']
    q_intent = answer['intent']
    keyterms = answer['key_terms']
    alt_queries = answer['alernate_queries']
    retrieved_date = answer['date_retrieved']

    retrieved_date = retrieved_date.strip()
    
    if retrieved_date.replace('.','').lower() != 'no date found':
        # print(f'Date was updated with: {retrieved_date}')
        retrieved_date = datetime.strptime(retrieved_date, '%d/%m/%Y')
        user_chat.update_project_date(retrieved_date)

    return user_chat.date_project,rephrased_query,alt_queries,keyterms,q_intent

def process_user_query(query,key_terms):

    system_instructions = """You are a specialized system that analyzes user queries and helps extract important information. Your task is to perform the following two actions:

	1.	Alpha Value Extraction:
	•	Analyze the user’s query and suggest an alpha value between 0 and 1 for hybrid search:
	•	A value close to 0 favors keyword-based search.
	•	A value close to 1 favors semantic-based search.
	•	Consider the complexity, intent, and specificity of the query when determining the value.
	2.	Synonym Generation:
	•	Based on the list of key terms provided, generate a list of synonyms for each key term.
	•	Ensure the synonyms are as close in meaning as possible, especially in the case of legal or technical terms, where slight differences in meaning might exist.

Output:

	•	Alpha Value for the query.
	•	List of Key Terms and Synonyms."""

    prompt = f"""Analyze the user's question: {query}.
    The key terms are: {key_terms}"""

    answer = call_gpt_api_with_single_prompt(instructions=dedent(system_instructions),
                                             prompt= dedent(prompt),
                                             response_format= user_query_format_alpha
                                             )

    answer = json.loads(answer)
    alpha_value = answer['alpha']
    synonyms = answer['synonyms']

    return alpha_value,synonyms

def update_date_conversation(user_chat,query='Text'):

    system_instructions = dedent('''Based on a given text, infer the date which is the question talking about. If no date is inferred, respond: "No date found", if yes, say the date in the following format: dd/mm/yyyy. Just answer with the date or the text. When no exact day can inferred, assume last day of the given month. If just the year is mention, say the last day of the year.
    If a date in the future is mentioned, return that no date was found. ''')

    query_send = f'''The text is the following:
    {query}'''

    retrieved_date = simple_llm_query(text=dedent(query_send),system_instruction=system_instructions)
    ## Clean date
    retrieved_date = retrieved_date.strip()
    

    if retrieved_date.replace('.','').lower() != 'no date found':
        # print(f'Date was updated with: {retrieved_date}')
        retrieved_date = datetime.strptime(retrieved_date, '%d/%m/%Y')
        user_chat.update_project_date(retrieved_date)
    # else:
    # print(retrieved_date)
    
    return user_chat.date_project

def send_long_message(bot, message, answer, user_chat, prompt_id = None,markup=None,max_length=4000):
    """
    Helper function to send a long message in chunks and store each reply.
    """
    # Split the answer into chunks with a maximum length of max_length
    chunks = smart_split(answer)
    messages_sent = []

    # Loop through each chunk, send it, and store it
    for i,chunk in enumerate(chunks):
        # Send the chunk as a reply to the original message
        if i == len(chunks)-1:
            markup_send = markup
        else:
            markup_send = None
        replied_msg = bot.reply_to(message, chunk,reply_markup = markup_send)

        # Store the reply message in the chat history
        user_chat.add_message(role='assistant', content=chunk, message_id=replied_msg.id, reply_to=message.id,prompt_id = prompt_id)

        # Append the sent message to the list of sent messages
        messages_sent.append(replied_msg)

    # Return the list of messages sent
    return messages_sent


    # add_message(self, role, content, message_id, reply_to = None, prompt_id=None)

def store_reply_message(bot, message, message_bot, user_chat):
    """
    Helper function to send and store a bot reply.
    """
    # Send the reply
    replied_msg = bot.reply_to(message, message_bot)

    # Store the reply message in the chat history
    user_chat.add_message(role='assistant', content=message_bot, message_id=replied_msg.id,reply_to = message.id)

    # Return the replied message (in case further processing is needed)
    return replied_msg

def store_sent_message(bot, message_bot_content, user_chat,prompt_id = None, reply_markup=None):
    """
    Helper function to send a bot message and store it in the chat history.
    
    :param bot: The bot instance.
    :param message: The original message object to which this message is related.
    :param message_bot_content: The content of the message that the bot will send.
    :param user_chat: The user's chat object to store the message in history.
    :param reply_markup: Optional. The reply markup (buttons, etc.) to be sent with the message.
    :return: The sent message object.
    """
    # Send the message using bot.send_message()
    chunks = smart_split(message_bot_content)
    messages_sent = []
    for i in chunks:

        sent_msg = bot.send_message(user_chat.id, i, reply_markup=reply_markup)

        # Store the sent message in the chat history
        user_chat.add_message(role='assistant', content=i, message_id=sent_msg.id,prompt_id = prompt_id)
        messages_sent.append(sent_msg)

    # Return the sent message object (in case further processing is needed)
    return messages_sent

def store_sent_image(bot, image_path, user_chat, caption=None, reply_markup=None, prompt_id=None):
    """
    Helper function to send an image and store it in the chat history.
    
    :param bot: The bot instance.
    :param image_path: Path to the image to be sent.
    :param user_chat: The user's chat object to store the image in history.
    :param caption: Optional. Caption to be sent with the image.
    :param reply_markup: Optional. The reply markup (buttons, etc.) to be sent with the message.
    :param prompt_id: Optional. ID of the prompt to store in the chat history.
    :return: The sent message object.
    """
    # Send the image using bot.send_photo()
    sent_msg = bot.send_photo(
        chat_id=user_chat.id, 
        photo=open(image_path, 'rb'), 
        caption=caption, 
        reply_markup=reply_markup
    )

    # Store the image path and details in the chat history
    # We're storing the image path as the 'text' here.
    user_chat.add_message(
        role='assistant', 
        content=f'Sent image: {image_path}', 
        message_id=sent_msg.id, 
        prompt_id=prompt_id
    )

    # Return the sent message object (in case further processing is needed)
    return sent_msg


# Define a function to check if a user is authorized
def is_user_authorized(user_id,username,authorized_users):
    if username is None:
        return str(user_id) in authorized_users    
    else:
        return username in authorized_users

def add_question(conversation,question):
    chat = {"role": "user", "content": question}

    conversation.append(chat)
    return conversation

def call_gpt_api_with_single_prompt(instructions, prompt, model="gpt-4o-2024-08-06", max_tokens=2500, response_format=None, img_path=None,detail='high'):
    """
    Sends a single message to GPT API with optional image input and retrieves the response.
    
    Parameters:
    - instructions: System instructions to set the context (e.g., "You are an AI assistant that analyzes tables").
    - prompt: User's message or query (e.g., "Please analyze the table in the image and provide a summary").
    - model: The GPT model to be used (default is "gpt-4o-2024-08-06").
    - max_tokens: Maximum number of tokens for the response (default is 2500).
    - response_format: Format of the response (e.g., "Rag_reponse"). Defaults to standard completion if not provided.
    - img_path: Optional path to an image file. If provided, the image will be included in the request.
    
    Returns:
    - The GPT answer object.
    """

    content = []
    dict_images = []
    # Create the messages list to send to GPT
    messages = [
        {"role": "system", "content": instructions}
    ]

    # If an image path is provided, encode and append it as a separate message
    if img_path:
        base64_image = encode_image(img_path)
        prompt_text = {'type':'text','text':dedent(prompt)}
        dic_images = {'type':'image_url','image_url':{'url': f"data:image/png;base64,{base64_image}",'detail':detail}}
        dict_images.append(dic_images)
        content.append(prompt_text)
        content.extend(dict_images)
        chat = {"role": "user", "content":content}

    else:
        chat = {"role": "user", "content":dedent(prompt)}
    
    messages.append(chat)
    
    try:
        if response_format == None:
            # Call GPT API using OpenAI's beta chat completions with parse
            response = openai.beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=max_tokens)
        else:
            # Call GPT API using OpenAI's beta chat completions with parse
            response = openai.beta.chat.completions.parse(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_format)

        # Extract and return the response content
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        print(f"Error during GPT API call: {e}")
        return None

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def add_question_context(conversation,question,context):

    if context == None:
        prompt = f"""Answer the following question: {question}.
There is no retrieved context for this question.

1. Review the previous messages and look for any relevant information or context.
2. If the question cannot be answered without additional information, say you don't have enough information to answer based on our previous discussions and retrieved context.
3. If the question is unrelated to your expertise (government financing and regulations for efficient buildings in Germany), kindly inform the user.
4. Reply in german, be polite trying to be friendly.
5. If applicable, mention any sources or prior context used in the response.

No additional context was found for this question."""
        chat = {"role": "user", "content":dedent(prompt)}
        ask = True
    else:
        ask = True
        prompt = f"""Answer the following question: {question}.
        Use the provided context. If the question cannot be answered with the context, say you can't answer the question with the retrieved context. Clarify if the questions is outside your expertise (government financing and regulations for efficient buildings in Germany).
        Review the previous messagges to check for any relevant information or context before answering.
        
        1. Reply in german, be polite trying to be friendly.
        2. Mention the source of your answer. Don't mention the IDs used, but the mention the source. For example: Richtlinie BEG EM,  Technische FAQ BEG, etc.
        3. Return the answer, stating the source and context used (Source, pages, IDs and literal text used) following the specified format.
        
        Retrieved Context:\n\n"""

        content = []
        dict_images = []
        n_images = 0
        for n,i in enumerate(context):
            context_i = f"Context {n} | ID: {i['id']} | {i['source']} Seiten: {i['pages']}\n{i['text']}"
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
            prompt_text = {'type':'text','text':dedent(prompt)}
            content.append(prompt_text)
            content.extend(dict_images)
            chat = {"role": "user", "content":content}
        else:
            chat = {"role": "user", "content":prompt}

        # print(f'Prompt sent to ChatGPT:\n{prompt}')

    conversation.append(chat)

    # print(f'Prompt\n{prompt}')
    
    return conversation,ask,prompt


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

def gen_markup_evaluate():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton("Ja ✅", callback_data="cb_yes"),
                InlineKeyboardButton("Nein ❌", callback_data="cb_no"))
    return markup

def gen_markup_evaluate_context():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton("Ja ✅", callback_data="cb_yes_context"),
                InlineKeyboardButton("Nein ❌", callback_data="cb_no_context"))
    return markup

def gen_markup_context():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Kontext anzeigen", callback_data = "show_context"))
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
    conversation,ask,prompt = add_question_context(conversation,question,context)
    if ask:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Change this to the desired model (e.g., "davinci" or "curie")
            messages=conversation,
            max_tokens=2000  # Adjust the maximum number of tokens for the response
        )
        answer = response.choices[0].message.content
        # print(f'Answer was obtained: {answer}')
        conversation = update_conversation(conversation,answer)
        return conversation,answer,prompt
    else:
        answer = 'Es wurde kein Kontext zu den gestellten Fragen gefunden. Bitte überprüfe die Frage oder formuliere sie gegebenenfalls um.'
        conversation = update_conversation(conversation,answer)
        return conversation,answer,prompt

# Define a function to ask a question to ChatGPT
def ask_question_with_context_json(conversation,question,context):
    # try:
    conversation,ask,prompt = add_question_context(conversation,question,context)
    if ask:
        
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Change this to the desired model (e.g., "davinci" or "curie")
            messages=conversation,
            max_tokens=4000,
            response_format = Rag_reponse  # Adjust the maximum number of tokens for the response
        )
        
        tokens_used = response.usage
        tokens_used = {"completion_tokens": tokens_used.completion_tokens, "prompt_tokens": tokens_used.prompt_tokens, "total_tokens": tokens_used.total_tokens}

        answer = clean_and_normalize_response(response.choices[0].message.content)
        
        answer = parse_rag_response_to_dict(answer)
        rag_answer = answer["answer"]
        context_used = answer["context_used"]
        # print(f'This is the context used: {context_used}')

        conversation = update_conversation(conversation,rag_answer)
        return conversation,rag_answer,prompt,tokens_used,context_used
    else:
        answer = 'Es wurde kein Kontext zu den gestellten Fragen gefunden. Bitte überprüfe die Frage oder formuliere sie gegebenenfalls um.'
        conversation = update_conversation(conversation,answer)
        context_used = []
        tokens_used = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        return conversation,answer,prompt,tokens_used,context_used


    
# Define a handler for the /start command
@bot.message_handler(commands=['getaccess'])
@track_active_user
@store_user_message
def get_access(message,user_chat):

    exist_user,allowed_user = is_user_allowed(db = sql_db_connector,
                                              telegram_username= user_chat.user,
                                              telegram_id= user_chat.id)
    
    print(f'user_name: {user_chat.user} exist {exist_user} allowed {allowed_user}')

    if allowed_user == 'False':
        message_bot = f'Willkommen {user_chat.name} bei {BOT_NAME}! Du darfst nicht chatten, bitte gib das Passwort ein, um Zugriff auf den Bot zu erhalten.'
        user_send_pwd.append(user_chat.user)
    elif exist_user == 'True' and allowed_user == 'True':
        message_bot = f"Willkommen {user_chat.name} bei {BOT_NAME}! Du darfst bereits chatten."
    if exist_user == 'False':
        # print('added user')
        add_user_to_db(sql_db_connector,
                       username=user_chat.user,
                       user_id=user_chat.id,
                       name=user_chat.name,
                       allowed=False)
    replied_msg = store_reply_message(bot,message,message_bot,user_chat)

# Define a handler for the /start command
@bot.message_handler(commands=['start'])
@track_active_user
@store_user_message
def send_welcome(message,user_chat):

    message_bot = f"Willkommen {user_chat.name} bei {BOT_NAME}! Schreib etwas, um mit dem Chatten zu beginnen."
    replied_msg = store_reply_message(bot,message,message_bot,user_chat)

# Define a handler for the /end command
@bot.message_handler(commands=['end'])
@track_active_user
@store_user_message
def send_end(message,user_chat):

    message_bot = f"Vielen Dank {user_chat.name}, dass du den {BOT_NAME} benutzt hast! Es war mir eine Freude, dir zu helfen."
    replied_msg = store_reply_message(bot,message,message_bot,user_chat)

# Define a handler for the /reset command
@bot.message_handler(commands=['reset'])
@track_active_user
@store_user_message
def send_reset(message,user_chat):

    message_bot = f"Deine Gesprächshistorie wird zurückgesetzt!"
    replied_msg = store_reply_message(bot,message,message_bot,user_chat)
    user_chat.reset_gpt_conversation_history()

# Define a handler for the /reset command
@bot.message_handler(commands=['cancel'])
@track_active_user
@store_user_message
def send_cancel(message,user_chat):

    message_bot = f"Vorgang abgebrochen!"
    replied_msg = store_reply_message(bot,message,message_bot,user_chat)

    if user_chat.id in user_send_pwd:
        user_send_pwd.remove(user_chat.id)


@bot.callback_query_handler(func=lambda call: True)
@track_active_user
def callback_query(call):
    global chat_manager
    user_id = call.message.chat.id
    mid = call.message.message_id
    user_name = call.message.from_user.username
    name_user = call.message.from_user.first_name
    text_message = call.message.text
    message_call = call.message

    user_chat = chat_manager.get_chat(user=user_name, first_name=name_user, user_id=user_id)

    text_answer = 'Deine Antwort war:'
    # print(f'Response eval: {call.data} - username: {user_name} name: {name_user} cid: {cid} mid: {mid}')
    if call.data == "cb_yes":
        bot.answer_callback_query(call.id, "Antwort ist Ja")
        bot.edit_message_text(chat_id=user_id, message_id=mid,
                          text=text_message, reply_markup=yes_markup())
        prompt_id = user_chat.edit_prompt_eval(mid = str(mid),
                                   value = 'good',
                                   column = 'evaluation')
        # bot.delete_message(cid,mid)
        # bot.send_message(cid,"Your answer was:", reply_markup=yes_markup())
    elif call.data == "cb_no":
        bot.answer_callback_query(call.id, "Antwort ist Nein")
        bot.edit_message_text(chat_id=user_id, message_id=mid,
                          text=text_message, reply_markup=no_markup())
        prompt_id = user_chat.edit_prompt_eval(mid = str(mid),
                                   value = 'bad',
                                   column = 'evaluation')
    elif call.data == "show_context":
        ## DO the search of the used context
        bot.answer_callback_query(call.id, "Kontext wird angezeigt")
        bot.edit_message_text(chat_id = user_id, message_id = mid,text = text_message,reply_markup = None)

        context,prompt_id = user_chat.extract_data_mid_prompt(mid = str(mid),column = 'context_used')

        if context:  # Check if context is not NULL or empty
            context_long_text = context  # This is the long TEXT stored in the database

            context_list = json.loads(context_long_text)
            send_long_message(bot = bot,
                              message=message_call,
                              answer='Kontext:',
                              user_chat= user_chat,
                              prompt_id=prompt_id)
            time.sleep(0.1)

            context_list_ids = [j.lower().replace(' ', '') for i in context_list for j in i['ids']]
            # context_list_ids = [i['id'].lower().replace(' ','') for i in context_list]
            records = get_data_ids(sql_db_connector,SQL_CHUNK_TABLE,context_list_ids)
            records = build_context(records,sql_db_connector,SQL_CHUNK_TABLE)

            for i in records:
                source_i = i['source']
                pages_i = i['pages']
                text_i = i['text']
                path_i = i['paths']

                message_i = f'Kontext: {source_i} - {pages_i} - {text_i[:10]} - {path_i}'

                message_send = f"Quelle: {i['source']} Seiten: {pages_i}\n{i['text']}"
                send_messages = store_sent_message(bot,message_send,user_chat,prompt_id)
                if len(path_i)>0:
                    for img in path_i:
                        img_send = store_sent_image(bot,img,user_chat,prompt_id=prompt_id)
                        print('send image!!!!!')

            time.sleep(0.1)
            send_long_message(bot = bot,
                    message=message_call,
                    answer='War der angezeigte Kontext hilfreich?',
                    user_chat= user_chat,
                    prompt_id=prompt_id,
                    markup=gen_markup_evaluate_context())
            

    elif call.data == 'cb_yes_context':
        bot.answer_callback_query(call.id, "Antwort ist Ja")
        bot.edit_message_text(chat_id=user_id, message_id=mid,
                    text=text_message, reply_markup=yes_markup())
        prompt_id = user_chat.edit_prompt_eval(mid = str(mid),
                            value = 'good',
                            column = 'context_eval')


    elif call.data == 'cb_no_context':
        bot.answer_callback_query(call.id, "Antwort ist Nein")
        bot.edit_message_text(chat_id=user_id, message_id=mid,
            text=text_message, reply_markup=no_markup())
        prompt_id = user_chat.edit_prompt_eval(mid = str(mid),
                            value = 'bad',
                            column = 'context_eval')


    elif call.data == 'cb_responded':
        bot.answer_callback_query(call.id,"Bereits beantwortet")
        # bot.delete_message(user_chat.id,mid)
        # bot.send_message(user_chat.id,"Your answer was:", reply_markup=no_markup())

@bot.message_handler(func=lambda message: message.reply_to_message)
@track_active_user
@store_user_message
def reply_msg(message,user_chat):

    comment = message.text

    prompt_id = user_chat.edit_prompt_eval(mid = str(message.reply_to_message.message_id),
                               value = comment,
                               column = "comment",
                               append = True)
    
    # Add prompt id to message
    update_data = {
        'prompt_id':prompt_id
    }
    conditions = {
        'chat_id':user_chat.id,
        'message_id':message.id
    }

    sql_db_connector.update_record(table_name=user_chat.messages_table,
                                   update_data=update_data,
                                   conditions=conditions)
    

# Define a handler for regular messages
@bot.message_handler(func=lambda message: True)
@track_active_user
@store_user_message
def echo_all(message,user_chat):
    global authorized_users
    message_rcvd = message.text
    message_id = message.message_id
    start_date = datetime.now()

    if user_chat.id in user_send_pwd:
        if message_rcvd == AUTH_TELEBOT_PWD:
            message_to_send = f'Richtiges Passwort! Du kannst jetzt mit {BOT_NAME} chatten!'
            add_user_to_db(sql_db_connector,
                username=user_chat.user,
                user_id=user_chat.id,
                name=user_chat.name,
                allowed=True)
            authorized_users = get_allowed_users(sql_db_connector)
            user_send_pwd.remove(user_chat.id)

        else:
            message_to_send = f'Falsches Passwort! Versuche es erneut! Sende /cancel, um die Anfrage zu beenden!'

        replied_msg = store_reply_message(bot,message,message_to_send,user_chat)

    elif is_user_authorized(user_chat.id,user_chat.user,authorized_users):

        # Event to signal the thread to stop
        stop_event = threading.Event()

        # Start a thread to keep sending 'typing' action
        typing_thread = threading.Thread(target=send_typing_action, args=(bot, user_chat.id, stop_event))
        typing_thread.start()
        # Get the message text
        conversation_history = user_chat.gpt_history.copy()

        input_text = message_rcvd

        # Check if the input text is too short
        if len(input_text.split()) < 2 or len(input_text) < 5:
            answer = "Deine Nachricht ist zu kurz, um verstanden zu werden. Bitte gib mehr Informationen an."
    
            replied_msg = store_reply_message(bot,message,answer,user_chat)
            return
        
        date_project,new_query,alt_queries,keyterms,q_intent = preprocess_user_query(user_chat,query=input_text)

        dates_files = get_dates_documents(date_project,date_database)
 
        date_query = display_date_to_user(date_project)

        if dates_files == None:

            print('No dates Found')
            stop_event.set()
            typing_thread.join()
            markup_send = None
            answer = f'Es wurden keine Dokumente für das angefragte Datum ({date_query}) gefunden. Die frühesten verfügbaren Dokumente stammen vom {MIN_DATE_DB}.'
            replied_msgs = send_long_message(bot, message, answer,user_chat,markup=markup_send)

        else:
            # print(f'DATES QUERY {dates_files}')

            results,ids_retrieved,filtered_ids = retrieve_context(
                vector_db_connector=pine_cone_connector,
                sql_connector=sql_db_connector,
                sql_table=SQL_CHUNK_TABLE,
                query=new_query,
                embed_task=EMBEDDING_MODEL_RET_TASK,
                max_results=MX_RESULTS,
                distance_threshold=DIST_THRESHOLD,
                retrieve_date_documents = dates_files,
                max_results_query = MX_RESULTS_QUERY,
                gen_prompts = alt_queries,
                keyterms = keyterms,
                vocab_table = SQL_VOCAB_BM25_TABLE,
                alpha_value=ALPHA_VALUE,
                embed_handler=embed_handler
                )

            results = build_context(results,sql_db_connector,SQL_CHUNK_TABLE)
            # Generate a response using ChatGPT
            conversation_history,answer,prompt,tokens_used,context = ask_question_with_context_json(conversation_history,input_text,results)

            answer += f' (Stand {date_query})'

            stop_event.set()
            typing_thread.join()
            markup_send = None
            if context:
                markup_send = gen_markup_context()
            replied_msgs = send_long_message(bot, message, answer,user_chat,markup=markup_send)
            related_mids = [i.id for i in replied_msgs]
            related_mids.append(message_id)
            end_date = datetime.now()
            
            quality_resp = store_sent_message(bot=bot,
                                            message_bot_content="War das eine gute Antwort?",
                                            user_chat=user_chat,
                                            reply_markup=gen_markup_evaluate())
            related_mids.append(quality_resp[0].id)
        
            prompt_id = user_chat.add_prompt(question = input_text,
                    context = results,
                    q_w_context = prompt,
                    answer = answer,
                    begin_date = start_date,
                    end_date = end_date,
                    messages_id = related_mids,
                    used_tokens = tokens_used,
                    context_used = context,
                    filtered_ids = filtered_ids,
                    retrieved_ids = ids_retrieved,
                    embed_model = EMBEDDING_MODEL,
                    chunk_size = MAX_TOKENS,
                    gen_prompts = alt_queries,
                    alpha_value = ALPHA_VALUE,
                    keyterms = keyterms,
                    q_intent = q_intent,
                    improved_query = new_query
                    )
            user_chat.reset_gpt_conversation_history(total = False)

    else:
        answer = f'Tut mir leid, du bist nicht berechtigt, mit diesem Bot zu interagieren. Verwende den Befehl /getaccess, um eine Erlaubnis anzufordern.'
        replied_msg = store_reply_message(bot,message,answer,user_chat)

#########################################################################
#
#       MAIN EXECUTION
#
#########################################################################

if __name__ == "__main__":
    manage_bot(bot)

   
