import datetime
import os
import csv

class Chat:
    def __init__(self, user,first_name,id,project_date = datetime.date.today().strftime("%d/%m/%Y"),file_gpt_prompts='',loc_conversations=''):
        self.user = user
        self.first_name = first_name
        self.id = id
        self.chat_history = []
        self.gpt_history = []
        self.name_user_id = f'{self.first_name}_{self.id}'
        self.file_prompts = file_gpt_prompts
        self.location_conversations = loc_conversations
        default_gpt_message = {"role": "system",
                               "content":"""
You are a helpful Retrieval-Augmented Generation (RAG) system specializing in answering questions related to government financing of efficient buildings in Germany. Your primary function is to clarify questions about the set of regulations and funding programs. Follow these guidelines to provide the best possible assistance:

1. Scope and Relevance:
• Focus exclusively on construction-related topics, specifically government financing and regulations for efficient buildings in Germany.
• If a query is unrelated, politely inform the user that you specialize in construction-related topics in Germany.

2. Language:
• Prefer responses in English and German.
• Use the informal ‘you’ (du) unless the user explicitly requests a formal tone.

3. Response Style:
• Provide precise and concise answers.
• Maintain a friendly and approachable tone.
• Include casual greetings and friendly messages to make interactions more human-like.

4. Sources and Credibility:
• Whenever possible, include the source of the information in your responses.
• Ensure the information is accurate and up-to-date, especially when referring to official documents and funding schemes.
• **If the user requests the exact wording from the source, you may quote directly from the context you have access to, provided it is relevant to the question.** 

5. Handling Unrelated Queries:
• If asked about unrelated topics, respond with:
“I am a specialized system for answering questions about government financing and regulations for efficient buildings in Germany. How can I assist you with that topic?”

6. Contextual Awareness:
• Always refer to previous user questions and your own past answers to maintain continuity in the conversation. Check prior exchanges for follow-up questions or references to previous topics.
• When a new question seems to reference earlier queries, incorporate the relevant context from earlier conversations into your response. This includes prior user questions, your previous answers, or key points mentioned in the conversation.

7. Clarifications:
• If the user’s question is unclear, try to rephrase the question using the context from previous exchanges.
• If the question still cannot be answered, ask the user for clarification, especially if it references previous topics not explicitly clear in the current query.

8. Efficient Context Management:
• Only include the most relevant information from past interactions to avoid unnecessary repetition. If previous interactions provide enough context to answer the question, rely on them instead of repeating the entire conversation.
""",
                               "context": None,
                               "timestamp": datetime.datetime.now(),
                               "chat_id": None,
                               "message_id": None
                               }
        self.gpt_history.append(default_gpt_message)
        self.project_date = project_date

    def update_project_date(self,project_date):
        self.project_date = project_date

    def reset_gpt_conversation_history(self, total=True, limit=10):
        limit = limit - 1
        if total:
            # Keep only the first message and discard the rest
            self.gpt_history = self.gpt_history[:1]  # Keep only the first message
        else:
            # Check if the limit is greater than the length of the history
            if len(self.gpt_history) <= limit + 1:
                # If the limit exceeds the length, leave the history as it is
                self.gpt_history = self.gpt_history  # Leave the history unchanged
            else:
                # Keep the first message and the last 'limit' messages
                self.gpt_history = self.gpt_history[:1] + self.gpt_history[-limit:]

    def add_message(self, role, content,chat_id,message_id, context=None, gpt_history=False):
        timestamp = datetime.datetime.now()
        if role not in ['assistant','system']:
            role = 'user'
        message = {
            "role": role,
            "content": content,
            "context": context,
            "timestamp": timestamp,
            "chat_id": chat_id,
            "message_id": message_id
        }
        if gpt_history:
            self.gpt_history.append(message)
        self.chat_history.append(message)
        self.save_to_text(message)

    def save_prompts(self,question,context,answer,evaluation,comment,mid_question,cid_question,mid_answer,cid_answer,name_user,user_name):
        prompt = {
            "question":question,
            "context":context,
            "answer":answer,
            "evaluation":evaluation,
            "comment":comment,
            "mid_question":mid_question,
            "cid_question":cid_question,
            "mid_answer":mid_answer,
            "cid_answer":cid_answer,
            "name_user":name_user,
            "user_name":user_name
        }
        # Check if the CSV file exists
        file_name = self.file_prompts
        file_exists = os.path.isfile(file_name)
        
        # Open the CSV file in append mode
        with open(file_name, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csvfile, fieldnames=prompt.keys())
            
            # If the file doesn't exist, write the header row
            if not file_exists:
                csv_writer.writeheader()
            
            # Write the data as a new row
            csv_writer.writerow(prompt)

    def edit_prompt_eval(self,cid_answer,mid_answer,value,column):
        # Open the CSV file in read mode
        filename = self.file_prompts
        with open(filename, 'r', newline='') as csvfile:
            # Create a CSV reader object
            csv_reader = csv.DictReader(csvfile)
            
            # Create a list to store rows
            rows = []
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Check if the values in column1 and column2 match the specified values
                if row['cid_answer'] == cid_answer and row['mid_answer'] == mid_answer:
                    # Update the value in the specified column
                    # print('Found match')
                    row[column] = value
                
                # Add the updated row to the list
                rows.append(row)
    
        # Write the updated rows back to the CSV file
        with open(filename, 'w', newline='') as csvfile:
            # Get the fieldnames from the first row of the CSV file
            fieldnames = rows[0].keys()
            
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header row
            csv_writer.writeheader()
            
            # Write the updated rows
            csv_writer.writerows(rows)

    def save_to_text(self, message):

        filename = self.location_conversations+f"/{self.name_user_id}_conversation_history.txt"
        sender = message['role'] if message['role'] != "assistant" else "assistant"
        line = f"[{message['timestamp']}] {sender}: {message['content']}\n"
        with open(filename, mode='a', encoding='utf-8') as file:
            file.write(line)

    def get_last_user_message(self):
        for message in reversed(self.messages):
            if message["role"] == "user":
                return message
        return None

    def get_last_assistant_message(self):
        for message in reversed(self.messages):
            if message["role"] == "assistant":
                return message
        return None

    def get_message_by_id(self, message_id):
        for message in self.messages:
            if message["id"] == message_id:
                return message
        return None
    
    def __str__(self):
        # return "username "+self.user + "first_name "+ self.first_name
        return f"Chat(user={self.user}, first_name={self.first_name}, project_date={self.project_date}, history={self.chat_history}, gpt_history={self.gpt_history})"

class ChatManager:
    def __init__(self,file_gpt_prompts,location_conversations):
        self.file_chat_gpt_prompts = file_gpt_prompts
        self.loc_conversations = location_conversations
        self.chats = {}

    def get_chat(self, user,first_name,id,project_date=datetime.date.today().strftime("%d/%m/%Y")):
        if user not in self.chats:
            self.chats[user] = Chat(user,first_name,id,project_date,self.file_chat_gpt_prompts,self.loc_conversations)
        return self.chats[user]