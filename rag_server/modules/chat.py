from datetime import datetime
from textwrap import dedent
import json
from config import SQL_MESSAGES_TABLE,SQL_PROMPTS_TABLE


class Chat:
    def __init__(self, user, name_user, user_id, db_connector,prompt_table,messages_table,date_project = datetime.now()):
        self.user = user
        self.id = user_id
        self.db_connector = db_connector  # Instance of MySQLDB for database operations
        self.chat_history = []  # In-memory cache for the chat history
        self.gpt_history = []  # In-memory GPT history
        self.name = name_user
        self.prompt_table = prompt_table
        self.messages_table = messages_table
        self.date_project = date_project

        # "content": "You are a helpful math tutor. Guide the user through the solution step by step."}

        # Initial GPT system message (as in your example)
        default_gpt_message = {"role": "system",
                               "content":dedent("""You are a specialized system focused on answering questions about government financing and regulations for efficient buildings in Germany.

	1.	For each question, structure your output in 2 parts:
	•	Answer: Provide a specific, accurate response to the user’s query, incorporating relevant information from the provided context. For regulatory topics, explain why or when something is necessary. Include the name of the source, don't mention IDs.
	•	Sources: Provide a list of the sources which were used to answer the question. Divide it into: text, source, id and pages.
	2.	Prioritize construction-related topics, particularly financing and regulations in Germany. If the question is outside your scope, politely clarify your area of expertise.
	3.	Provide concise, accurate, and specific answers. Always include explanations for your reasoning when applicable.
	4.	Maintain context and continuity by referencing past interactions. If a question is linked to a previous one, use relevant past information, even if the new question is phrased differently.
	5.	If a question is unclear, rephrase it for clarity or ask for more information, using prior exchanges when possible to guide your clarification.
	6.	When applicable, include sources to back up your answers, ensuring they are up-to-date. If the user requests exact wording, quote directly from the source if it is available in the provided context.
                                                
    **Important: Always reply in german, be polite and use greetings to the user, trying to be friendly.""")
}
        self.gpt_history.append(default_gpt_message)

    def add_message(self, role, content, message_id, reply_to = None, prompt_id=None,device = None):
        """
        Add a message to the Messages table and cache it in memory.
        """
        timestamp = datetime.now()
        message = {
            "role": role,
            "message": content,
            "date": timestamp,
            "chat_id": self.id,
            'reply_message_id':reply_to,
            "message_id": message_id,
            "prompt_id": prompt_id,
            "device":device
        }

        # Add to the in-memory history
        self.chat_history.append(message)

        # Insert into the database
        self.db_connector.insert_record(self.messages_table,message)

    def update_project_date(self,project_date):
        self.date_project = project_date

    def add_prompt(self, question, context, q_w_context, answer,
                   begin_date, end_date, messages_id,used_tokens,
                   context_used,filtered_ids,retrieved_ids,
                   chunk_size,embed_model,keyterms,
                   q_intent,improved_query,
                   alpha_value = None,
                   gen_prompts = None,
                   evaluation = None, comment = None,
                   setting = "Default",
                   device = None):
        """
        Save a GPT prompt to the database and associate with chat.
        """
        if gen_prompts:
            gen_prompts = json.dumps(gen_prompts,ensure_ascii=False)
        if keyterms:
            keyterms = json.dumps(keyterms,ensure_ascii=False)

        context_string = ''
        if context is not None:
            for i in context:
                    text = f'{i['source']} pages: {i['pages']}\n{i['text'][:100]}\n{i['paths']}\n'
                    space = '======== ======== ======== ======== ========\n'
                    context_string+=text+space

        q_w_r_context = f'{question}'
        if not context_used:
            context_used_db = None  # Return None to store as NULL in the database
        else:
            context_used_db = json.dumps(context_used,ensure_ascii=False)  # Convert the list to a JSON string
            q_w_r_context += '\nContext:\n'
            for i in context_used:
                q_w_r_context+=f'Source: {i['source']}\n{i['text']}\n'

        if filtered_ids:
            filtered_ids = json.dumps(filtered_ids,ensure_ascii=False)
        if retrieved_ids:
            retrieved_ids = json.dumps(retrieved_ids,ensure_ascii=False)
        
        filtered_ids,retrieved_ids


        prompt_entry = {
            "device": device,
            "chat_id": self.id,
            "question": question,
            "context": context_string,
            "question_w_context":q_w_context,
            "answer": answer,
            "evaluation": evaluation,
            "comment": comment,
            "query_date": self.date_project,
            "begin_date": begin_date,
            "end_date": end_date,
            "completion_tokens" : used_tokens['completion_tokens'],
            'prompt_tokens': used_tokens['prompt_tokens'],
            'total_tokens': used_tokens['total_tokens'],
            'context_used': context_used_db,
            'context_ids':filtered_ids,
            'context_ids_total':retrieved_ids,
            'alternate_prompts': gen_prompts,
            'embedding_model': embed_model,
            'chunk_size': chunk_size,
            'alpha_value': alpha_value,
            'improved_query':improved_query,
            'query_intent':q_intent,
            'keyterms':keyterms,
            'setting':setting
        }

        # Insert the prompt into the Prompts table
        prompt_id = self.db_connector.insert_record(self.prompt_table, prompt_entry)

        # add prompt_id -> messages
        if len(messages_id) > 0:
            for message_id in messages_id:
                # Prepare the data to update the message record with the prompt_id
                update_data = {
                    "prompt_id": prompt_id,
                    "device": device
                }
                update_filter = {
                    "chat_id" : self.id,
                    "message_id" : str(message_id)
                }
                # Update the message record in the Messages table with the prompt_id
                self.db_connector.update_record(self.messages_table, update_data,update_filter)  # Assuming 'id' is the primary key column



        prompt_user = {
            "role":"user",
            "content":q_w_r_context
        }
        prompt_assistant = {
            "role":"assistant",
            "content":answer
        }

        self.gpt_history.append(prompt_user)
        self.gpt_history.append(prompt_assistant)

        return prompt_id

    def reset_gpt_conversation_history(self, total=True, limit=10):
        """
        Reset the GPT history in memory, but keep it in the database.
        """
        if total:
            self.gpt_history = self.gpt_history[:1]  # Keep the first system message
        else:
            if len(self.gpt_history) <= limit + 1:
                self.gpt_history = self.gpt_history
            else:
                self.gpt_history = self.gpt_history[:1] + self.gpt_history[-limit:]


    """
    Change the function to read the gpt history from the database, only keep the last 10 records
    """
    def load_chat_history_from_db(self):
        """
        Load the chat history from the database into memory.
        """
        query = f"SELECT * FROM Messages WHERE chat_id = %s ORDER BY timestamp ASC"
        self.db_connector.cursor.execute(query, (self.id,))
        rows = self.db_connector.cursor.fetchall()

        # Load chat history into memory
        for row in rows:
            self.chat_history.append({
                "role": row["role"],
                "content": row["content"],
                "context": row["context"],
                "date": row["date"],
                "chat_id": row["chat_id"],
                "message_id": row["message_id"],
                "prompt_id": row.get("prompt_id")
            })

    def load_gpt_history(self, limit=None):
        """
        Load GPT history from the Prompts table, with an option to limit the number of entries.
        For each prompt:
        - The question with context is treated as a 'user' message.
        - The answer is treated as an 'assistant' message.
        """
        query = f"""
            SELECT question_w_context, answer FROM Prompts 
            WHERE chat_id = %s 
            ORDER BY query_date DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        # Fetch the GPT-related history from the Prompts table
        self.db_connector.cursor.execute(query, (self.id,))
        rows = self.db_connector.cursor.fetchall()

        # Reverse to maintain chronological order and load into in-memory history
        for row in reversed(rows):
            # Add the user's question with context
            self.gpt_history.append({"role": "user", "content": row[0]})
            # Add the assistant's answer
            self.gpt_history.append({"role": "assistant", "content": row[1]})
        
    def edit_prompt_eval(self,mid,value,column,append = False):

        # Get prompt to edit -> Retrieve value from messages_table
        values = ['prompt_id','device']
        conditions = {
            "chat_id" : self.id,
            "message_id" : mid
        }
        results = self.db_connector.get_record(table_name = self.messages_table,
                                                 values = values,
                                                 conditions = conditions)
        prompt_id = results[0]
        device = results[1]
        # print(f'Prompt id found {prompt_id}')
        conditions_prompt = {'prompt_id':prompt_id,
                             'device':device}
        values = {column:value}
        self.db_connector.update_record(table_name = self.prompt_table,
                                        update_data = values,
                                        conditions = conditions_prompt,
                                        append = append)

        # print('editting values')
        return prompt_id,device
    
    def extract_data_mid_prompt(self,mid,column):

        # Get prompt to edit -> Retrieve value from messages_table
        values = ['prompt_id','device']
        conditions = {
            "chat_id" : self.id,
            "message_id" : mid
        }
        results = self.db_connector.get_record(table_name = self.messages_table,
                                                 values = values,
                                                 conditions = conditions)
        prompt_id = results[0]
        device = results[1]
        # Get prompt to edit -> Retrieve value from messages_table
        values = [column]
        conditions = {
            "prompt_id" : prompt_id,
            "device": device
        }
        column_value = self.db_connector.get_record(table_name = self.prompt_table,
                                            values = values,
                                            conditions = conditions)
        
        column_value = column_value[0]

        return column_value,prompt_id,device
    
    def retrieve_last_prompt_id(self):

        values = ['prompt_id','device']
        conditions = {"chat_id": self.id}


        records = self.db_connector.get_records(table_name = self.prompt_table,
                                      values = values,
                                      conditions = conditions)
        last_record = records[-1]
        prompt_id = last_record['prompt_id']
        device = last_record['device']

        return prompt_id,device

    
class ChatManager:
    def __init__(self, db_connector,messages_table = SQL_MESSAGES_TABLE, prompt_table = SQL_PROMPTS_TABLE):
        self.db_connector = db_connector
        self.chats = {}
        self.messages_table = messages_table
        self.prompt_table = prompt_table

    def get_chat(self, user, first_name, user_id):
        """
        Get or create a new chat session.
        """
        if user_id not in self.chats:
            new_chat = Chat(user, first_name, user_id, self.db_connector,prompt_table = self.prompt_table,messages_table=self.messages_table)
            # new_chat.load_chat_history_from_db()  # Load chat history from the database
            self.chats[user_id] = new_chat
        return self.chats[user_id]