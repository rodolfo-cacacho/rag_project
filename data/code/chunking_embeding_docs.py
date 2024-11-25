from document_cleaning import pre_clean_text,post_clean_document
from langchain_core.documents import Document
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
import json
import numpy as np
import unicodedata
import re
import math
from collections import defaultdict,Counter
import spacy
import nltk
from nltk.corpus import stopwords
import openai
import base64
from pydantic import BaseModel
from textwrap import dedent
import pandas as pd
from nltk import ngrams
from collections import Counter
import Levenshtein
from tqdm import tqdm

# Load German spaCy model
nlp = spacy.load('de_core_news_lg')

API_KEY_CGPT = "sk-proj-FrNMaLZT7tgBftIDPuBOT3BlbkFJFND9eOXtUMeCyWxplFlV"
openai.api_key = API_KEY_CGPT


# Load German stopwords and the German spaCy model
nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))
domain_stopwords = ['sowie','überblick','beziehungsweise','verschieden','beispiel','gemäß']

# Config and initialization
config = {
    'user': 'root',
    'password': 'admin123',
    'host': '127.0.0.1'
}
db_name = 'data_rag'
table_documents_name = 'table_documents'
table_retrieval_name = 'embedding_table'
table_retrieval_schema = {
'id': 'varchar(10) NOT NULL PRIMARY KEY',
'content': 'longtext NOT NULL',
'metadata': 'longtext NOT NULL'
}

table_pages_clean_schema = {
'id': 'INT NOT NULL PRIMARY KEY',
'clean_text':'longtext NOT NULL',
'placeholders':'longtext',
'correct':'BOOLEAN DEFAULT FALSE'
}

table_vocabulary_schema = {
'id': 'INT NOT NULL PRIMARY KEY',
'word': 'varchar(100) NOT NULL',
'idf': 'FLOAT NOT NULL',
'id_docs': 'longtext NOT NULL'
}

def process_metadata_csv(metadata_csv_path):

    df_codes = pd.read_csv(metadata_csv_path)
    df_codes['type_key'] = df_codes['type_key'].astype('int16')
    df_codes['file_key'] = df_codes['file_key'].astype('int16')
    # Convert document_date to datetime if it's not already
    df_codes['date_c'] = pd.to_datetime(df_codes['date'], format="%d/%m/%Y", errors='coerce', dayfirst=True)
    # Sort by date to ensure the most recent appears last within each type_key
    df_codes = df_codes.sort_values(['type_key', 'date_c'], ascending=[True, False])
    # Identify the most recent document within each type_key
    df_codes['most_recent'] = df_codes.groupby('type_key')['date_c'].transform('max') == df_codes['date_c']

    return df_codes

def load_documents_pages(sql_con,table = table_documents_name):

    records = sql_con.get_all_records_as_dict(table)
    docs = []
    sub_docs = []
    act_doc = None
    for n,i in enumerate(records):
        page = i.copy()
        page['content'] = pre_clean_text(i['content'])
        pdf_name = i['pdf_name']
        if act_doc != pdf_name and n > 0:
            act_doc = pdf_name
            docs.append(sub_docs)
            sub_docs = []
            sub_docs.append(page)
        elif n == len(records)-1:
            sub_docs.append(page)
            docs.append(sub_docs)
        else:
            act_doc = pdf_name
            sub_docs.append(page)

    return docs


# Try another tokenizer/embedding model: 
# 1. danielheinz/e5-base-sts-en-de
# 2. jinaai/jina-embeddings-v2-base-de

def semantic_chunking_embedding(docs, main_id, df_codes,sql_con,vector_db_con,max_tokens=300,chars = 200,table_name = table_retrieval_name,table_schema = table_retrieval_schema):
    # Maximum number of tokens in a chunk
    tokenizer = Tokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-de")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    new_docs = []
    ids = []
    dict_docs = []
    n = 0

    print(f'Pages: {len(docs)}')
    for i_doc in docs:
        metadata = i_doc['metadata']
        metadata = unicodedata.normalize('NFC', metadata)
        metadata = json.loads(metadata)
        # metadata = {k: v.encode('latin1').decode('utf-8') if isinstance(v, str) else v for k, v in metadata.items()}

        # Get the PDF name from the metadata
        pdf_name = metadata["source"]

        # Find the row where the file matches the PDF name
        matched_row = df_codes[df_codes['file'] == pdf_name]

        # Check if any rows were found
        if not matched_row.empty:
            # Extract type_key and file_key from the matched row (assuming there's only one match)
            metadata["type_key"] = int(matched_row["type_key"].iloc[0])
            metadata["file_key"] = int(matched_row["file_key"].iloc[0])
        else:
            # If no match found, set type_key and file_key to 0
            metadata["type_key"] = 0
            metadata["file_key"] = 0

        # Extract and clean the title from the metadata
        content = i_doc['content']
        content = unicodedata.normalize('NFC', content)

        chunks = splitter.chunks(content)
        for j, chunk in enumerate(chunks):
            doc_id = f'{main_id}.{n}'
            pre_id = f'{main_id}.{n-1}' if n > 0 else ""
            post_id = f'{main_id}.{n+1}' if j < len(chunks) - 1 or (i_doc != docs[-1] or j != len(chunks) - 1) else ""
            
            updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
            updated_metadata['id'] = doc_id
            updated_metadata['pre_id'] = pre_id
            updated_metadata['post_id'] = post_id
            
            doc = Document(page_content=chunk, metadata=updated_metadata)
            new_docs.append(doc)
            ids.append(doc_id)
            n += 1

    # After final chunk is processed, fix the last document's post_id
    if new_docs:
        new_docs[-1].metadata['post_id'] = ""

    for i,doc in enumerate(new_docs):
        pdf_name = doc.metadata['source']
        title = pdf_name.replace('.pdf', '')
        chunk = doc.page_content
        new_chunk = f'''{title}\n\n{chunk}'''
        doc.page_content = new_chunk
        doc_id = doc.metadata['id']
        dict = {'id':doc_id,
                'content':chunk,
                'metadata':str(doc.metadata)}
        dict_docs.append(dict)

    sql_con.create_table(table_name=table_name,
                         schema=table_schema)
    
    sql_con.insert_many_records(table_name=table_name,records=dict_docs)
    
    vector_db_con.add_documents([new_docs],[ids])
    # print(f'{len(added_ids)} ids were added.')

    return new_docs

def merge_pages(pages):

    merged_doc = ""

    for i,page_l in enumerate(pages):
        page = page_l[0]
        type = page_l[1]
        if i < len(pages) - 1:
            next_type = pages[i+1][1]
            page = page.strip()
            match = re.search(r'-\s*$', page)
            if match:
                page = re.sub(r'-\s*$', '', page)
                merged_doc += page
            else:
                next_page = pages[i+1][0].strip()
                match_act_page = re.search(r'[,.!?;:]\s*$', page)
                match_nxt_page = re.match(r'^[\d\-\•\*\(\)a-z]\.?', next_page)
                match_nxt_page_cont = re.match(r'^[a-zäöüß]{2,}', next_page)
                match_act_page_capt = re.search(r'\b[A-ZÄÖÜ][a-zäöüß]*\b$',page)
                match_nxt_page_capt = re.match(r'^[A-ZÄÖÜ][a-zäöüß]*',next_page)
                match_next_page_nr = re.match(r'^Nr\.',next_page)
                if (type == 'Table' or next_type == 'Table') or match_next_page_nr:
                    merged_doc += f'{page}\n\n'
                elif match_act_page or match_nxt_page or (match_act_page_capt and match_nxt_page_capt):
                    merged_doc += f'{page}\n'
                else:
                    merged_doc += f'{page} '
                
        else:
            merged_doc += page

    return merged_doc

def load_act_pages(act_pages,pages,clean_version = False):
    
    if clean_version:
        key_extract = 'clean_text'
    else:
        key_extract = 'content'

    if len(act_pages)>1:
        act_pages_text = []

        for i in act_pages:
            act_pages_text.append((pages[i][key_extract],pages[i]['type']))

        act_page_content = merge_pages(act_pages_text)
        act_page_content = act_page_content.split()

    else:
        act_page_content = pages[act_pages[0]][key_extract].split()

    return act_page_content


def merge_metadata(metadata_list):
    pages = []
    types = []
    paths = []
    for i in metadata_list:
        pages.append(i['page'])
        types.append(i['type'])
        paths.append(i['path'])

    metadata_dict = metadata_list[0]
    pages = sorted(set(pages))
    metadata_dict['page'] = ', '.join(map(str, pages))
    # 1. Handling the types (Text or Table)
    if 'Text' in types and 'Table' in types:
        type_output = 'Text + Table'
    elif 'Text' in types:
        type_output = 'Text'
    elif 'Table' in types:
        type_output = 'Table'
    else:
        type_output = 'Unknown'

    metadata_dict['type'] = type_output

    # 2. Filtering and handling file paths
    filtered_paths = [path for path in paths if path]  # Remove empty paths
    metadata_dict['path'] = filtered_paths
    
    return metadata_dict


class Placeholder(BaseModel):
    placeholder_type: str
    text: str

class Clean_text(BaseModel):
    clean_text: str
    placeholders: list[Placeholder]

class Clean_text_simple(BaseModel):
    clean_text: str

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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


def verify_chunk(text,placeholders):
    placeholder_counts = Counter(item['placeholder_type'] for item in placeholders)
    for placeholder_type, count in placeholder_counts.items():
        count_in_text = text.count(f"[{placeholder_type}]")
        
        # Check for mismatch between counts
        if count != count_in_text:
            return False
    
    # If no mismatches are found, return True
    return True       

instructions_cleaning_chunk = """This is a system designed to clean and structure German-language texts related to regulatory guidelines, green financial aid for buildings, and similar topics. The objective is to improve semantic chunking and embedding accuracy by removing noise and irrelevant text while maintaining placeholders for essential information. This cleaning protocol ensures that only meaningful content is embedded, while retaining critical placeholders for terms, dates, and references commonly found in regulatory documents.

Instructions

	1.	General Cleaning and Noise Removal:
	•	Remove redundant punctuation (e.g., multiple periods ..., excessive spaces) and any symbols that don’t add meaning.
	•	Preserve hyphens within compound words relevant to the document’s context, such as Energie-Effizienz.
	•	Discard formal or repetitive phrases that don’t contribute to semantic meaning, like introductory notes (e.g., “Wichtiger Hinweis”, “Bitte beachten Sie”).
    •   Don't change the text, don't complete it and don't rephrase it, just clean what could add noise to the embeddings by removing it. You are given chunks that may not be complete.
    •   While cleaning, retain line breaks where they logically divide sections or distinct topics within the text. If line breaks appear to separate unrelated content, such as a new topic or regulation clause, keep them intact. Only remove line breaks that do not contribute to the clarity or logical flow, such as multiple consecutive breaks or breaks within sentences.
    •   For the Table of contents/Inhalt or similar, keep the line breaks. 
	2.	Placeholder Masking (in German):
	•	For specific content types, replace with German-labeled placeholders. Follow this format:
	•	[Datum] – For dates (e.g., 01.01.2021, 20.06.2023, not Datum/Daten).
	•	[Wert] – For numeric values and measurements (e.g., 15 kWh, 50%, €1000).
	•	[Version] – For version details (e.g., Version 2.0, v3.1, ignore when not accompanied by a number).
	•	[Gesetz] – For legal references, such as § 35a, Artikel 27.
	•	[Email] – For email addresses (e.g., kontakt@unternehmen.de).
	•	[URL] – For URLs (e.g., http://www.beispiel.de).
	•	[Abschnitt] – For section or chapter headings (e.g., Abschnitt 2.3, Kapitel 4.5).
	•	[Telefon] – For phone numbers (e.g., 0800 123456).
    •   [Jahr] - For years (e.g, 2020, 2 019, etc.)
	•	Additional Flexibility: The model should recognize other relevant parts and add placeholders for these as needed to enhance retrieval accuracy. Special cases could be equations, insitutions, etc.
	3.	Output Requirements:
	•	Placeholder List: Provide a List showing each placeholder and its corresponding original text, for instance, [Datum]: ["01.01.2021"].
	•	Return Clean Text: Supply the final text with placeholders in place, optimized for semantic embedding.
    •   Make sure to match the number of placeholders in the text as in the list!!!"""

instructions_cleaning_chunk_simple = """This is a fallback system designed to minimally clean German-language texts related to regulatory guidelines, green financial aid for buildings, and similar topics. The objective is to remove basic noise and redundant text elements without changing the text’s meaning or structure. This protocol ensures that text remains as close to the original as possible while making it more suitable for embedding.

Fallback Cleaning Instructions

	1.	General Cleaning and Noise Removal:
	•	Remove excessive whitespace and redundant spaces within words; retain only single spaces between words.
	•	Replace multiple consecutive punctuation marks with a single instance:
	•	Periods: Replace ... with .
	•	Commas: Replace ,,, with ,
	•	Exclamation Marks: Replace !!! with !
	•	Question Marks: Replace ??? with ?
	•	Remove any isolated special characters that add no meaning (e.g., @#$%, unless they’re part of structured information like emails or URLs).
	•	Hyphen Preservation: Keep hyphens within compound words meaningful to the document’s context (e.g., “Energie-Effizienz”).
	2.	Line Break Retention and Removal:
	•	Retain line breaks if they logically separate different sections, topics, or distinct clauses within the text. This applies especially in cases where line breaks divide unrelated content, such as a new topic or regulation clause.
	•	Remove only redundant line breaks, such as multiple consecutive breaks, or breaks that disrupt the flow within a single sentence.
	•	For sections like “Inhalt” or “Table of Contents,” keep all line breaks to preserve readability.
	3.	Character Clean-Up and Final Check:
	•	Ensure no leading or trailing whitespace remains in the text.
	•	Double-check that any punctuation cleaning or line break adjustment hasn’t altered the logical flow or meaning of the text.

Output Requirements:

	•	Return Clean Text: Provide the final text with noise removed."""

def clean_pages(df_codes,sql_con,table_pages_clean,table_pages_clean_schema=table_pages_clean_schema,retries_max = 3,only_recents = True,repeat_incorrect = False):

    # Prepare data
    sql_con.create_table(table_pages_clean,table_pages_clean_schema)
    processed_list = sql_con.get_all_records_as_dict(table_pages_clean)
    processed_list = [{'id': entry['id'], 'correct': entry['correct']} for entry in processed_list]
    processed_list = {entry['id']: entry['correct'] for entry in processed_list}


    docs = load_documents_pages(sql_con)

    for i,doc in enumerate(docs):
        metadata = doc[0]['metadata']
        metadata = json.loads(metadata)
        pdf_name = metadata["source"]
        matched_row = df_codes[df_codes['file'] == pdf_name]
        if not matched_row.empty:
            # Extract type_key and file_key from the matched row (assuming there's only one match)
            recent = matched_row["most_recent"].iloc[0]
        else:
            # If no match found, set type_key and file_key to 0
            recent = False
        print(f'{i}-{pdf_name} recent:{recent}')

        if not only_recents or (only_recents and recent):
            print(f'Processing... {len(doc)} Pages')
            chunks_processed = []
            # Check if already exists or it is incorrectly processed!!

            for num_page,page in enumerate(doc):
                id_page = page['id']

                if id_page not in processed_list or (id_page in processed_list and  processed_list.get(id_page) == 0 and repeat_incorrect):
                
                    retries = 0
                    check = False
                    page_content = page['content']
                    prompt_chunk = f'Process the following chunk:\n{page_content}'

                    while not check:
                        if retries > 0:
                            prompt_chunk_s = f'Make sure to match the number of placeholders in the text and the list returned. This is a retry. {prompt_chunk}'
                            print(f'Retrying {retries}')
                        else:
                            prompt_chunk_s = prompt_chunk
                        
                        answer = call_gpt_api_with_single_prompt(instructions=instructions_cleaning_chunk,
                                                                prompt= prompt_chunk_s,
                                                                max_tokens=8000,
                                                                response_format=Clean_text)
                        answer = json.loads(answer)
                        check = verify_chunk(answer['clean_text'],answer['placeholders'])

                        if check:
                            print(f'Page {num_page} done Processed')
                            answer['correct'] = True
                            answer['id'] = page['id']
                            answer['placeholders'] = json.dumps(answer['placeholders'])
                            chunks_processed.append(answer)
                        else:
                            retries+=1
                            if retries >= retries_max:
                                prompt_chunk_simple = f'Process the following chunk:\n{page_content}'
                                answer = call_gpt_api_with_single_prompt(instructions=instructions_cleaning_chunk_simple,
                                                                         prompt= prompt_chunk_simple,
                                                                         max_tokens=8000,
                                                                         response_format=Clean_text_simple)
                                answer = json.loads(answer)
                                page_content = answer['clean_text']
                                wrong_answer = {'clean_text':page_content,
                                                'placeholders': None,
                                                'correct': False,
                                                'id':page['id']}
                                chunks_processed.append(wrong_answer)
                                check = True
                                print(f'Page {num_page} done Simple')
                else:
                    print(f'Page {num_page} done')
                    

            # Store in DB
            sql_con.insert_many_records(table_pages_clean,chunks_processed)

    print('Listo')


def semantic_chunking_v2(docs,doc_type, main_id, df_codes,sql_con,vector_db_con,max_tokens=300,chars = 200,table_name = table_retrieval_name,table_schema = table_retrieval_schema,pdf_type_chunked = ['Technische FAQ BEG EM']):
    # Maximum number of tokens in a chunk
    pages = docs
    tokenizer = Tokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-de")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    new_docs = []
    ids = []
    dict_docs = []

    if doc_type in pdf_type_chunked:

        for i_chunk,chunk in enumerate(pages):

            chunk_content = chunk['content'].split()
            word_count = len(chunk_content)

            metadata = chunk['metadata']
            metadata = json.loads(metadata)
            pdf_name = metadata["source"]
            matched_row = df_codes[df_codes['file'] == pdf_name]
            # Check if any rows were found
            if not matched_row.empty:
                # Extract type_key and file_key from the matched row (assuming there's only one match)
                metadata["type_key"] = int(matched_row["type_key"].iloc[0])
                metadata["file_key"] = int(matched_row["file_key"].iloc[0])
            else:
                # If no match found, set type_key and file_key to 0
                metadata["type_key"] = 0
                metadata["file_key"] = 0

            doc_id = f'{main_id}.{i_chunk}'
            last_id = f'{main_id}.{len(pages) - 1}'
                
            updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
            updated_metadata['id'] = doc_id
            updated_metadata['last_id'] = last_id
            updated_metadata['word_count'] = word_count
                
            doc = Document(page_content=chunk['content'], metadata=updated_metadata)
            new_docs.append(doc)
            ids.append(doc_id)

    else:

        pages_list = []
        for i in pages:
            i['content'] = unicodedata.normalize('NFC',i['content'])
            page = i['content']
            pages_list.append(page)

        merged_doc = merge_pages(pages_list)

        # CHUNKING #
        chunks = splitter.chunks(merged_doc)

        act_page = [0]  # Track the last matched page index

        for i_chunk,chunk in enumerate(chunks):
            metadata = []
            chunk_content = chunk.split()
            word_count = len(chunk_content)

            act_page_content = load_act_pages(act_page,pages)
            
            # print(f'Len chunk {len(chunk_content)} Len page {len(act_page_content)}')
            while len(chunk_content) > len(act_page_content):
                act_page.append(act_page[-1]+1)
                act_page_content = load_act_pages(act_page,pages)

            # print(f'Chunk {i}:\n{chunk_content}\n\nPage {act_page}:\n{act_page_content}')

            # print(f'Chunk {i} Page {act_page}')
            # We will remove words from the page content only if they match sequentially with the chunk content
            match_index = 0  # Keeps track of how many words from the chunk have matched

            # Start comparing words from the chunk with words from the page content
            while match_index < len(chunk_content) and match_index < len(act_page_content):
                if chunk_content[match_index] == act_page_content[match_index]:
                    # If the words match, move to the next word in both the chunk and the page
                    match_index += 1
                else:
                    # If the words don't match, stop the matching process
                    break

            # print(f'Removed {match_index} - Chunk size {len(chunk_content)}')

            act_page_content = act_page_content[match_index:]  # Remove matched words from the start of act_page_content
            for page in act_page:
                metadata.append(pages[page]['metadata'])

            if len(act_page_content) > 0:
                new_pages_act = act_page.copy()
                for i,page in enumerate(act_page):
                    if i == len(act_page) - 1:
                        pages[page]['content'] = ' '.join(act_page_content)
                    else:
                        pages[page]['content'] = ''
                        new_pages_act.pop(0)
                act_page = new_pages_act
            else:
                for i in act_page:
                    pages[i]['content'] = ''
                    act_page = [act_page[-1]+1]

            metadata = [json.loads(unicodedata.normalize('NFC',x)) for x in metadata]
            metadata = merge_metadata(metadata)
            pdf_name = metadata["source"]
            matched_row = df_codes[df_codes['file'] == pdf_name]
            # Check if any rows were found
            if not matched_row.empty:
                # Extract type_key and file_key from the matched row (assuming there's only one match)
                metadata["type_key"] = int(matched_row["type_key"].iloc[0])
                metadata["file_key"] = int(matched_row["file_key"].iloc[0])
            else:
                # If no match found, set type_key and file_key to 0
                metadata["type_key"] = 0
                metadata["file_key"] = 0

            doc_id = f'{main_id}.{i_chunk}'
            last_id = f'{main_id}.{len(chunks) - 1}'
                
            updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
            updated_metadata['id'] = doc_id
            updated_metadata['last_id'] = last_id
            updated_metadata['word_count'] = word_count
                
            doc = Document(page_content=chunk, metadata=updated_metadata)
            new_docs.append(doc)
            ids.append(doc_id)

    for i,doc in enumerate(new_docs):
        title = doc.metadata['doc_type']
        chunk = doc.page_content
        new_chunk = f'''{title}\n\n{chunk}'''
        doc.page_content = new_chunk
        doc_id = doc.metadata['id']
        dict = {'id':doc_id,
                'content':chunk,
                'metadata':str(doc.metadata)}
        dict_docs.append(dict)

    sql_con.create_table(table_name=table_name,
                         schema=table_schema)
    
    print(f'SQL to be inserted: {len(dict_docs)}')
    # for n,i in enumerate(dict_docs):
    #     print(f'{n}\nContent: {i['content']}')
    #     print(f'Metadata: {i['metadata']}')
    sql_con.insert_many_records(table_name=table_name,records=dict_docs)
    
    # print(f'Pinecone to be inserted: {len(new_docs)}|{len(ids)}')
    # vector_db_con.add_documents([new_docs],[ids])

    return new_docs

def build_vocabulary(cleaned_chunks,chunks_ids):
    """Builds vocabulary from a list of cleaned chunks and initializes DF counts."""
    vocabulary = {}
    df = defaultdict(int)

    for i,chunk in enumerate(cleaned_chunks):
        unique_terms = set(chunk)  # Count each term once per document for DF
        for token in unique_terms:
            if token not in vocabulary:
                vocabulary[token] = {"index": len(vocabulary), "word": token, "idf": 0,"docs":[chunks_ids[i]]}  # Initialize term with index and IDF placeholder
            else:
                vocabulary[token]["docs"].append(chunks_ids[i])
            df[token] += 1  # Increment DF for this term in the document

    return vocabulary, df

def compute_idf(vocabulary, df, N):
    """Computes inverse document frequency (IDF) for each term and updates vocabulary."""
    for token, data in vocabulary.items():
        df_count = df[token]
        idf_value = math.log((N / (df_count + 1)))  # Add 1 to avoid division by zero
        vocabulary[token]["idf"] = idf_value  # Update vocabulary with IDF
    return vocabulary

def compute_term_frequencies(cleaned_chunks, vocabulary):
    """Computes term frequencies (TF) for each document and document lengths."""
    tf_matrix = []
    doc_lengths = []

    for chunk in cleaned_chunks:
        tf = defaultdict(int)
        for token in chunk:
            if token in vocabulary:
                tf[token] += 1
        tf_matrix.append(tf)
        doc_lengths.append(len(chunk))
    
    return tf_matrix, doc_lengths

def compute_bm25(tf_matrix, vocabulary, doc_lengths, avg_dl, k1=1.5, b=0.75):
    """Computes the BM25 scores for each document."""
    bm25_matrix = []
    for i, tf in enumerate(tf_matrix):
        bm25 = np.zeros(len(vocabulary))
        doc_len = doc_lengths[i]
        for token, freq in tf.items():
            if token in vocabulary:
                index = vocabulary[token]["index"]
                idf = vocabulary[token]["idf"]
                tf_component = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_len / avg_dl)))
                bm25[index] = tf_component * idf
        bm25_matrix.append(bm25)
    return bm25_matrix

def create_pinecone_updates(bm25_matrix, chunk_ids):
    """Creates a list of dictionaries in Pinecone update format, excluding empty sparse vectors."""
    updates = []
    for i, bm25 in enumerate(bm25_matrix):
        # Get non-zero values (important for sparse vectors)
        non_zero_indices = np.nonzero(bm25)[0]
        non_zero_values = bm25[non_zero_indices]

        # Create the dictionary in the Pinecone update format
        update_data = {
            "id": chunk_ids[i],
            "sparse_values": {
                "indices": non_zero_indices.tolist(),
                "values": non_zero_values.tolist()
            }
        }
        updates.append(update_data)
    
    return updates

def process_chunks_bm25(cleaned_chunks, chunk_ids):
    """
    Main function to process the cleaned chunks using BM25:
    1. Builds the vocabulary with IDF.
    2. Computes TF, DF, and BM25 scores.
    3. Creates a list of dictionaries with Pinecone update format.
    """
    # Step 1: Build vocabulary and document frequencies (DF)
    vocabulary, df = build_vocabulary(cleaned_chunks,chunk_ids)

    # Step 2: Compute inverse document frequency (IDF) and update vocabulary
    N = len(cleaned_chunks)  # Total number of documents
    vocabulary = compute_idf(vocabulary, df, N)

    # Step 3: Compute term frequencies (TF) and document lengths
    tf_matrix, doc_lengths = compute_term_frequencies(cleaned_chunks, vocabulary)

    # Step 4: Compute average document length
    avg_dl = sum(doc_lengths) / N

    # Step 5: Compute BM25 scores for each document
    bm25_matrix = compute_bm25(tf_matrix, vocabulary, doc_lengths, avg_dl)

    # Step 6: Create Pinecone update format
    updates = create_pinecone_updates(bm25_matrix, chunk_ids)

    return vocabulary, updates

def preprocess_text(text):
    """Remove line breaks and standalone special characters from text."""
    # Replace line breaks with a space
    text = text.replace('\n', ' ')
    
    # Remove standalone special characters surrounded by spaces
    text = re.sub(r'\s[-/]\s', ' ', text)  # Example for " - " and " / "
    
    # Remove extra spaces that may have been created
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_urls_and_entities(text):
    # Extract URLs
    url_pattern = r'https?://\S+|www\.\S+'
    urls = re.findall(url_pattern, text)
    for url in urls:
        text = text.replace(url, '')  # Remove URLs from text

    # Extract standard entities with spaCy
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    for entity in entities:
        text = text.replace(entity, '')  # Remove entities from text

    return text, urls, entities

def isolate_number_unit_terms(text):
    """Find and remove terms with a number followed by a unit (e.g., '525 kWh/m2')."""
    number_unit_pattern = r'\b\d+(?:[\.,]\d+)?\s*(?:kwh/m2|°c|m2|g|kg|l|ml|w|kw|v|a)\b'
    number_unit_terms = re.findall(number_unit_pattern, text, re.IGNORECASE)
    for term in number_unit_terms:
        text = text.replace(term, '')  # Remove terms with numbers and units

    return text, number_unit_terms

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_chapters_and_sections(text):
    """
    Identifies chapter and section numbers in a given text, such as '3.03', '6.1.4.3', etc.
    
    Parameters:
    - text: str, input text to search for sections.
    
    Returns:
    - List of chapter/section numbers found in the text.
    """
    # Regular expression pattern to capture chapter/section numbers
    section_pattern = r'\b\d+(?:\.\d+)+\b'
    
    # Find all matches of the pattern in the text
    sections = re.findall(section_pattern, text)
    for term in sections:
        text = text.replace(term,'')
    
    return text,sections

def find_norms(text):
    """
    Identifies norms like DIN, ISO, EN, etc., with their associated names, numbers, and variants.
    
    Parameters:
    - text: str, input text to search for norms.
    
    Returns:
    - List of norms found in the text.
    """
    # Regular expression pattern to capture norms
    norm_pattern = r'\b(?:DIN|ISO|EN)(?:\s+[A-Z]{1,3})?\s+\d{3,5}(?:[-‑]\d+)?\b'
    
    # Find all matches of the pattern in the text
    norms = re.findall(norm_pattern, text, re.IGNORECASE)
    for term in norms:
        text = text.replace(term,'')
    
    return text,norms

def find_short_terms(text):
    """
    Identifies short terms like 'WG' or 'NWG' that are 2-3 characters long and uppercase.
    
    Parameters:
    - text: str, input text to search for short terms.
    
    Returns:
    - Tuple: modified text (with short terms removed), list of short terms found.
    """
    short_term_pattern = r'\b[A-Z]{2,3}\b'
    short_terms = re.findall(short_term_pattern, text)
    
    for term in short_terms:
        text = text.replace(term, '')
        
    return text, short_terms

def clean_chunk(text):
    # Step 1: Preprocess text to remove line breaks and standalone special characters
    text = preprocess_text(text)

    # Step 2: Extract URLs, entities, and number-unit terms, removing them from text
    text,sections = find_chapters_and_sections(text)
    text,norms = find_norms(text)
    text,short_terms = find_short_terms(text)
    text, number_unit_terms = isolate_number_unit_terms(text)
    text, urls, entities = extract_urls_and_entities(text)
    # Step 3: Clean text to remove unwanted characters and extra spaces
    text = clean_text(text)

    # Step 4: Tokenize, remove stopwords, and perform lemmatization using spaCy
    doc = nlp(text)
    
    # Step 5: Filter out stopwords and short words (words with length <= 2)
    cleaned_tokens = [
        token.lemma_.lower() for token in doc 
        if token.text.lower() not in german_stopwords and len(token.text) > 2
    ]

    # Step 6: Extend the tokens with extracted URLs, entities, and number-unit terms
    cleaned_tokens.extend(short_terms)
    cleaned_tokens.extend(sections)
    cleaned_tokens.extend(norms)
    cleaned_tokens.extend(urls)
    cleaned_tokens.extend(entities)
    cleaned_tokens.extend(number_unit_terms)
    cleaned_tokens = [token.lower() for token in cleaned_tokens]

    # Step 7: Return the final list of tokens
    return cleaned_tokens

def calculate_bm_25_score(sql_con,vector_db_con,table_chunks,table_vocabulary,schema_table_vocabulary=table_vocabulary_schema,print_n = 1000):

    records = sql_con.get_all_records_as_dict(table_chunks)
    cleaned_chunks = []
    chunk_ids = []
    for n,i in enumerate(records):
        chunk_ids.append(i['id'])
        cleaned_chunks.append(clean_chunk(i['content']))
        if (n+1) % print_n == 0:
            print(f'{n+1}/{len(records)} processed...')


    vocabulary, vector_updates = process_chunks_bm25(cleaned_chunks,chunk_ids)
    print(f'Vocabulary: {len(vocabulary)}\nChunks {len(vector_updates)}')


    ids = []
    for n,i in enumerate(vector_updates):
        ids.append(records[n]['id'])
        i['metadata'] = json.loads(records[n]['metadata'].replace("'",'"'))
        i['content'] = records[n]['content']

    print('Upserting to Pinecone')
    vector_db_con.add_documents([vector_updates],[ids])
    sql_con.delete_table(table_vocabulary)
    sql_con.create_table(table_vocabulary,schema_table_vocabulary)

    dict_vocabulary = []
    # print(vocabulary)
    for term,word in vocabulary.items():
        dict = {'id':word['index'],
                'word':word['word'],
                'idf':word['idf'],
                'id_docs':(",".join(f"'{item}'" for item in word['docs']))}
        dict_vocabulary.append(dict)
        
    print('Upserting to MySQL')

    sql_exec = sql_con.insert_many_records(table_vocabulary,dict_vocabulary)
    print(f'{len(sql_exec)} were inserted')

    return records

def load_documents_pages_clean(sql_con,table,table_clean_documents):

    records = sql_con.get_all_records_as_dict(table)
    clean_records = sql_con.get_all_records_as_dict(table_clean_documents)
    processed_list = {entry['id']: entry for entry in clean_records}

    docs = []
    sub_docs = []
    act_doc = None
    for n,i in enumerate(records):
        id = i['id']
        pdf_name = i['pdf_name']
        if id in processed_list:
            vals = processed_list.get(id)
            i.update(vals)
        
        if act_doc != pdf_name and n > 0:
            act_doc = pdf_name
            docs.append(sub_docs)
            sub_docs = []
            sub_docs.append(i)
        elif n == len(records)-1:
            sub_docs.append(i)
            docs.append(sub_docs)
        else:
            act_doc = pdf_name
            sub_docs.append(i)

    return docs

def replace_placeholders_sequentially(chunks, replacements):
    # Track the replacements and remove each after use
    remaining_replacements = replacements.copy()
    updated_chunks = []

    # Define regex pattern to detect placeholders
    placeholder_pattern = re.compile(r'\[(\w+)\]')  # Matches patterns like [Wert], [Datum], etc.

    # Iterate through each chunk
    for chunk in chunks:
        # Track changes to the chunk
        modified_chunk = chunk

        # Find placeholders in the current chunk
        placeholders_in_chunk = placeholder_pattern.findall(chunk)

        # Process each placeholder in order of appearance in the chunk
        for placeholder_type in placeholders_in_chunk:
            # Find the next available replacement of the current placeholder type
            replacement_idx = next((i for i, rep in enumerate(remaining_replacements) if rep["placeholder_type"] == placeholder_type), None)
            
            # Replace if a matching replacement was found
            if replacement_idx is not None:
                replacement_text = remaining_replacements[replacement_idx]["text"]
                
                # Replace the first occurrence of the placeholder
                modified_chunk = modified_chunk.replace(f"[{placeholder_type}]", replacement_text, 1)
                
                # Remove the used replacement to prevent reuse
                del remaining_replacements[replacement_idx]
        
        # Add modified chunk to the result list
        updated_chunks.append(modified_chunk)
    
    return updated_chunks



def semantic_chunking_v3(docs,doc_type, clean_version,main_id, df_codes,sql_con,max_tokens=300,table_name = table_retrieval_name,table_schema = table_retrieval_schema,pdf_type_chunked = ['Technische FAQ BEG EM']):
    # Maximum number of tokens in a chunk
    pages = docs
    tokenizer = Tokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-de")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    clean_texts = []
    placeholders = []
    clean_docs = []
    embedding_docs = []
    ids = []
    metadata_clean = []

    if doc_type in pdf_type_chunked:

        for i_chunk,chunk in enumerate(pages):

            if clean_version:
                chunk_text = chunk['clean_text']
                placeholder = chunk['placeholders']
                if placeholder is not None:
                    placeholder = json.loads(placeholder)
                    placeholders.extend(placeholder)
                chunk_text = unicodedata.normalize('NFC',chunk_text)
            else:
                chunk_text = chunk['content']
                chunk_text = unicodedata.normalize('NFC',chunk_text)

            
            chunk_content = chunk_text.split()
            word_count = len(chunk_content)

            metadata = chunk['metadata']
            metadata = json.loads(metadata)
            pdf_name = metadata["source"]
            matched_row = df_codes[df_codes['file'] == pdf_name]
            # Check if any rows were found
            if not matched_row.empty:
                # Extract type_key and file_key from the matched row (assuming there's only one match)
                metadata["type_key"] = int(matched_row["type_key"].iloc[0])
                metadata["file_key"] = int(matched_row["file_key"].iloc[0])
            else:
                # If no match found, set type_key and file_key to 0
                metadata["type_key"] = 0
                metadata["file_key"] = 0

            doc_id = f'{main_id}.{i_chunk}'
            last_id = f'{main_id}.{len(pages) - 1}'
            ids.append(doc_id)

            updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
            updated_metadata['id'] = doc_id
            updated_metadata['last_id'] = last_id
            updated_metadata['word_count'] = word_count

            if clean_version:
                print(f'Function to get normal data back')
                clean_text = chunk['content']
                embedding_doc = Document(page_content= clean_text, metadata=updated_metadata)
                embedding_docs.append(embedding_doc)

                clean_text = chunk['clean_text']
                clean_texts.append(clean_text)
                metadata_clean.append(updated_metadata)

            else:
                clean_text = chunk['content']
                doc = Document(page_content= clean_text, metadata=updated_metadata)
                embedding_docs.append(doc)

    else:

        pages_list = []
        for i in pages:

            if clean_version:
                key_version = 'clean_text'
                page = i['clean_text']
                placeholder = i['placeholders']
                if placeholder is not None:
                    placeholder = json.loads(placeholder)
                    placeholders.extend(placeholder)
                page = unicodedata.normalize('NFC',page)
            else:
                key_version = 'content'
                page = i['content']
                page = unicodedata.normalize('NFC',page)

            pages_list.append(page)

        merged_doc = merge_pages(pages_list)

        # CHUNKING #
        chunks = splitter.chunks(merged_doc)

        act_page = [0]  # Track the last matched page index

        for i_chunk,chunk in enumerate(chunks):
            metadata = []
            chunk_content = chunk.split()
            word_count = len(chunk_content)

            act_page_content = load_act_pages(act_page,pages,clean_version)
            # print(f'Len chunk {len(chunk_content)} Len page {len(act_page_content)}')
            while len(chunk_content) > len(act_page_content):
                act_page.append(act_page[-1]+1)
                act_page_content = load_act_pages(act_page,pages,clean_version)


            # print(f'Chunk {i}:\n{chunk_content}\n\nPage {act_page}:\n{act_page_content}')

            # print(f'Chunk {i} Page {act_page}')
            # We will remove words from the page content only if they match sequentially with the chunk content
            match_index = 0  # Keeps track of how many words from the chunk have matched

            # Start comparing words from the chunk with words from the page content
            while match_index < len(chunk_content) and match_index < len(act_page_content):
                if chunk_content[match_index] == act_page_content[match_index]:
                    # If the words match, move to the next word in both the chunk and the page
                    match_index += 1
                else:
                    # If the words don't match, stop the matching process
                    break

            # print(f'Removed {match_index} - Chunk size {len(chunk_content)}')

            act_page_content = act_page_content[match_index:]  # Remove matched words from the start of act_page_content
            for page in act_page:
                metadata.append(pages[page]['metadata'])

            if len(act_page_content) > 0:
                new_pages_act = act_page.copy()
                for i,page in enumerate(act_page):
                    if i == len(act_page) - 1:
                        pages[page][key_version] = ' '.join(act_page_content)
                    else:
                        pages[page][key_version] = ''
                        new_pages_act.pop(0)
                act_page = new_pages_act
            else:
                for i in act_page:
                    pages[i][key_version] = ''
                    act_page = [act_page[-1]+1]

            metadata = [json.loads(unicodedata.normalize('NFC',x)) for x in metadata]
            metadata = merge_metadata(metadata)
            pdf_name = metadata["source"]
            matched_row = df_codes[df_codes['file'] == pdf_name]
            # Check if any rows were found
            if not matched_row.empty:
                # Extract type_key and file_key from the matched row (assuming there's only one match)
                metadata["type_key"] = int(matched_row["type_key"].iloc[0])
                metadata["file_key"] = int(matched_row["file_key"].iloc[0])
            else:
                # If no match found, set type_key and file_key to 0
                metadata["type_key"] = 0
                metadata["file_key"] = 0

            doc_id = f'{main_id}.{i_chunk}'
            last_id = f'{main_id}.{len(chunks) - 1}'
                
            updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
            updated_metadata['id'] = doc_id
            updated_metadata['last_id'] = last_id
            updated_metadata['word_count'] = word_count

            if clean_version:
                text_embed = chunk
                embedding_doc = Document(page_content= text_embed, metadata=updated_metadata)
                embedding_docs.append(embedding_doc)
                
                clean_texts.append(chunk)
                metadata_clean.append(updated_metadata)

            else:
                clean_text = chunk
                doc = Document(page_content= clean_text, metadata=updated_metadata)
                embedding_docs.append(doc)

            ids.append(doc_id)


    # Embedding

    dict_docs = []
    for i,doc in enumerate(embedding_docs):
        chunk = doc.page_content
        doc_id = doc.metadata['id']
        dict = {'id':doc_id,
                'content':chunk,
                'metadata':json.dumps(doc.metadata)}
        dict_docs.append(dict)

    table_name_embedding = table_name+'_embedding'
    print(f'SQL to be inserted (Embedding): {len(dict_docs)}')
    sql_con.create_table(table_name=table_name_embedding,
                        schema=table_schema)
    sql_con.insert_many_records(table_name=table_name_embedding,records=dict_docs)
    
    # Clean docs - Retrieval
    
    if clean_version:
        dict_docs = []

        clean_chunks = replace_placeholders_sequentially(clean_texts,placeholders)

        for i,doc in enumerate(clean_chunks):
            chunk = doc
            metadata_chunk = metadata_clean[i]
            doc_id = metadata_chunk['id']
            word_count_chunk = len(chunk.split())
            metadata_chunk['word_count'] = word_count_chunk
            dict = {'id':doc_id,
                    'content':chunk,
                    'metadata':json.dumps(metadata_chunk)}
            dict_docs.append(dict)

        print(f'SQL to be inserted (Clean): {len(dict_docs)}')
        sql_con.create_table(table_name=table_name,
                            schema=table_schema)
        sql_con.insert_many_records(table_name=table_name,records=dict_docs)

    else:

        print(f'SQL to be inserted (Clean): {len(dict_docs)}')
        sql_con.create_table(table_name=table_name,
                        schema=table_schema)
        sql_con.insert_many_records(table_name=table_name,records=dict_docs)

    return clean_docs

def upsert_vector_bm25_calc(sql_con,vector_db_con,table_chunks_text,table_chunks_embs,table_vocabulary,schema_table_vocabulary=table_vocabulary_schema,print_n = 1000):

    records_text = sql_con.get_all_records_as_dict(table_chunks_text)
    records_embed = sql_con.get_all_records_as_dict(table_chunks_embs)
    # Creating a dictionary for fast lookup in records_embed by id
    embed_dict = {record['id']: record['content'] for record in records_embed}

    # Merging records_text with content from records_embed
    records = []
    for record in records_text:
        # Rename 'content' to 'content_bm25' in the main list
        merged_record = {
            'id': record['id'],
            'content_bm25': record['content'],
            'metadata': record['metadata'],
            'content': embed_dict.get(record['id'], '')  # Add content from records_embed
        }
        records.append(merged_record)

    del(records_embed,records_text,embed_dict)

    cleaned_chunks = []
    chunk_ids = []
    for n,i in enumerate(records):
        chunk_ids.append(i['id'])
        cleaned_chunks.append(clean_chunk(i['content_bm25']))
        if (n+1) % print_n == 0:
            print(f'{n+1}/{len(records)} processed...')

    vocabulary, vector_updates = process_chunks_bm25(cleaned_chunks,chunk_ids)
    print(f'Vocabulary: {len(vocabulary)}\nChunks {len(vector_updates)}')

    ids = []
    for n,i in enumerate(vector_updates):
        ids.append(records[n]['id'])
        i['metadata'] = json.loads(records[n]['metadata'].replace("'",'"'))
        i['content'] = records[n]['content']

    print('Upserting to Pinecone')
    vector_db_con.add_documents(vector_updates,ids)
    sql_con.delete_table(table_vocabulary)
    sql_con.create_table(table_vocabulary,schema_table_vocabulary)

    dict_vocabulary = []
    # print(vocabulary)
    for term,word in vocabulary.items():
        dict = {'id':word['index'],
                'word':word['word'],
                'idf':word['idf'],
                'id_docs':(",".join(f"'{item}'" for item in word['docs']))}
        dict_vocabulary.append(dict)
        
    print('Upserting to MySQL')

    sql_exec = sql_con.insert_many_records(table_vocabulary,dict_vocabulary)
    print(f'{len(sql_exec)} were inserted')

    return records


def semantic_chunking(sql_con,
                      table_chunks_name,
                      df_code_path,
                      max_tokens = 500,
                      output_dir = '.',
                      table_documents_name = table_documents_name,
                      table_chunks_schema = table_retrieval_schema,
                      tokenizer_embed_model = "jinaai/jina-embeddings-v2-base-de",
                      pdf_type_chunked = ['Technische FAQ BEG EM']):
    
    """
    Full semantic chunking of documents (Only chunking, no embedding)

    Paramenters:
    - sql_con (MySQLDB): MySQL Db connector
    - table_documents_name (str): Table name with all documents and pages
    - table_chunks_name (str): Table name where chunks will be stored
    - df_code_path (str): Path of df_codes, contains metadata of all documents
    - max_tokens (int): Max number of tokens for the semantic chunking
    - tokenizer_embed_model (str): Name of the embedding model from Huggingface used as base for the semantic chunking; default: jinaai/jina-embeddings-v2-base-de
    - pdf_type_chunked (list): List of str with names of documents already 'chunked'

    Returns:
    - str: The cleaned text
    """
    chunks_total_upserted = 0

    docs = load_documents_pages(sql_con = sql_con,
                                table = table_documents_name)
    
    # docs = docs[:n_limit]
    
    df_codes = process_metadata_csv(df_code_path)

    # Maximum number of tokens in a chunk
    tokenizer = Tokenizer.from_pretrained(tokenizer_embed_model)
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    sql_con.delete_table(table_chunks_name)
    sql_con.create_table(table_chunks_name,table_chunks_schema)

    print(f'Documents: {len(docs)}')

    for idoc,doc in enumerate(docs,start=1):

        # Get document type using first page
        chunks_list = []
        doc_type = doc[0]['pdf_type']
        # print(f"Document {doc[0]['pdf_name']} : Pages {len(doc)}")

        if doc_type not in pdf_type_chunked:
        
            # Gotta merge the document here:::
            pages_list = []

            for page in doc:
                pages_list.append((page['content'],page['type']))

            merged_doc = merge_pages(pages_list)

            # print(f'Merged Pages {len(pages_list)}\n{merged_doc}\n\n')

            # CHUNKING #
            chunks = splitter.chunks(merged_doc)
            # print(f'Total Chunks {len(chunks)}')
            print(f"Document {doc[0]['pdf_name']} : Pages {len(doc)} : Chunks {len(chunks)}")


            act_page = [0]  # Track the last matched page index

            for ichunk,chunk in enumerate(chunks):
                # print(f'Chunk {ichunk}\nActual pages {act_page}.')
                metadata = []
                chunk_content = chunk.split()
                word_count_chunk = len(chunk_content)

                act_page_content = load_act_pages(act_page,doc)
                
                while len(chunk_content) > len(act_page_content):
                    act_page.append(act_page[-1]+1)
                    act_page_content = load_act_pages(act_page,doc)
                # if len(chunk_content)==len(act_page_content):
                #     print('MATCH')
                # print(f'Chunk_words:{chunk_content}\nAct_page_words:{act_page_content}\nAct pages: {act_page}\n')

                # We will remove words from the page content only if they match sequentially with the chunk content
                match_index = 0  # Keeps track of how many words from the chunk have matched

                # Start comparing words from the chunk with words from the page content
                while match_index < len(chunk_content) and match_index < len(act_page_content):
                    if chunk_content[match_index] == act_page_content[match_index]:
                        # If the words match, move to the next word in both the chunk and the page
                        match_index += 1
                    else:
                        # If the words don't match, stop the matching process
                        break
                
                # Remove matched words from the start of act_page_content
                act_page_content = act_page_content[match_index:]

                for page in act_page:
                    metadata.append(doc[page]['metadata'])

                if len(act_page_content) > 0:
                    new_pages_act = act_page.copy()
                    for i,page in enumerate(act_page):
                        if i == len(act_page) - 1:
                            doc[page]['content'] = ' '.join(act_page_content)
                        else:
                            doc[page]['content'] = ''
                            new_pages_act.pop(0)
                    act_page = new_pages_act
                else:
                    for i in act_page:
                        doc[i]['content'] = ''
                    act_page = [act_page[-1]+1]

                metadata = [json.loads(x) for x in metadata]
                metadata = merge_metadata(metadata)
                pdf_name = metadata["source"]
                matched_row = df_codes[df_codes['file'] == pdf_name]
                # Check if any rows were found
                if not matched_row.empty:
                    # Extract type_key and file_key from the matched row (assuming there's only one match)
                    metadata["type_key"] = int(matched_row["type_key"].iloc[0])
                    metadata["file_key"] = int(matched_row["file_key"].iloc[0])
                else:
                    # If no match found, set type_key and file_key to 0
                    metadata["type_key"] = 0
                    metadata["file_key"] = 0

                doc_id = f'{idoc}.{ichunk}'
                last_id = f'{idoc}.{len(chunks) - 1}'
                    
                updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
                updated_metadata['id'] = doc_id
                updated_metadata['last_id'] = last_id
                updated_metadata['word_count'] = word_count_chunk
                    
                chunk_dict = {'id':doc_id,
                              'content':chunk,
                              'metadata':json.dumps(updated_metadata,ensure_ascii=False)}
                
                # Upsert to DB!!!
                chunks_list.append(chunk_dict)
            
        else: 

            print(f"Document {doc[0]['pdf_name']} : Pages {len(doc)} : Chunks {len(chunks)}")
            

            for ichunk,chunk in enumerate(doc):

                chunk_content = chunk['content'].split()
                word_count = len(chunk_content)

                metadata = chunk['metadata']
                metadata = json.loads(metadata)
                pdf_name = metadata["source"]
                matched_row = df_codes[df_codes['file'] == pdf_name]
                # Check if any rows were found
                if not matched_row.empty:
                    # Extract type_key and file_key from the matched row (assuming there's only one match)
                    metadata["type_key"] = int(matched_row["type_key"].iloc[0])
                    metadata["file_key"] = int(matched_row["file_key"].iloc[0])
                else:
                    # If no match found, set type_key and file_key to 0
                    metadata["type_key"] = 0
                    metadata["file_key"] = 0

                doc_id = f'{idoc}.{ichunk}'
                last_id = f'{idoc}.{len(doc) - 1}'
                    
                updated_metadata = metadata.copy()  # Copy the metadata to avoid modifying the original
                updated_metadata['id'] = doc_id
                updated_metadata['last_id'] = last_id
                updated_metadata['word_count'] = word_count

                chunk_dict = {'id':doc_id,
                              'content':chunk['content'],
                              'metadata':json.dumps(updated_metadata,ensure_ascii=False)}
                    
                # Upsert to DB!!!
                chunks_list.append(chunk_dict)


        ids_upserted = sql_con.insert_many_records(table_name = table_chunks_name,
                                    records = chunks_list,
                                    overwrite = False)

        chunks_total_upserted += len(ids_upserted)

    sql_con.export_table(table_name = table_chunks_name,
                        export_format = 'json',
                        output_dir = output_dir)

    return chunks_total_upserted


def get_table_data_as_dict(sql_con=None, json_path=None, table_name=None):
    """
    Retrieves table data as a list of dictionaries from either a SQL connection or a JSON file.

    :param sql_con (MySQLDB): An object with a `get_all_records_as_dict` method for database access.
    :param json_path (str): Path to the JSON file. Used if sql_con is None.
    :param table_name (str): Name of the table to retrieve from the database, or ignored if using JSON.
    :return: A list of dictionaries containing the table data.
    """
    if sql_con is not None:
        # Use sql_con's method to retrieve data from the database
        records = sql_con.get_all_records_as_dict(table_name)
        print(f"Data retrieved from database table '{table_name}'.")
    elif json_path is not None:
        # Read data from JSON file and convert it to a list of dictionaries
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"Data retrieved from JSON file at '{json_path}'.")
    else:
        raise ValueError("Either sql_con or json_path must be provided.")
    
    return records

def store_vocabulary(vocabulary,schema, sql_con=None, table_name=None, json_path=None):
    """
    Stores the vocabulary in MySQL or as a JSON file.

    Args:
        vocabulary (dict): The vocabulary dictionary with word IDs and metadata.
        sql_con (MySQLDB): MySQL connection object.
        table_name (str): Name of the table to store the vocabulary in MySQL.
        json_path (str): Path to store the vocabulary as a JSON file.
    """
    vocab_data = [
        {
            "id": data["id"],
            "word": word,
            "idf": data["idf"],
            "synonyms": json.dumps(data["synonyms"], ensure_ascii=False)
        }
        for word, data in vocabulary.items()
    ]

    if sql_con:
        sql_con.create_table(table_name, schema)
        sql_con.insert_many_records(table_name, vocab_data)
    elif json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)


def embedding_bm25_calculation(table_name,table_store,json_path = None,sql_con = None):
    """
    Function to clean and calculate the embeddings

    :param table_name (str):
    """

    records = get_table_data_as_dict(sql_con=sql_con,
                                     json_path=json_path,
                                     table_name=table_name)
    
    voc_analyzer = WordFrequencyAnalyzer()
    
    for irecord,record in enumerate(records):

        id = record['id']
        content = record['content']
        metadata = json.loads(record['metadata'])
        clean_content,extracted_values = post_clean_document(content)
        record['clean_content'] = clean_content
        record['extracted_values'] = extracted_values
        tokens_content = extract_tokens_lemm_stop(clean_content)
        record['tokens'] = tokens_content
        new_tokens = clean_extracted_values_to_tokens(extracted_values)
        record['tokens'].extend(new_tokens)
        tokens = record['tokens']
        voc_analyzer.update_words(tokens)

    # Example vocabulary
    vocabulary = [word for word,freq in voc_analyzer.word_frequencies.items()]

    # Define exclusions
    exclude_words = {"co2", "co", "o2"}
    ignore_norms_pattern = r"DIN EN ISO \d+"

    # Group synonyms
    synonym_groups = group_synonyms_with_prefix_check(
        vocabulary,
        threshold=0.95,
        exclude_words=exclude_words,
        ignore_norms_pattern=ignore_norms_pattern
    )

    # Clean Vocabulary
    vocabulary = merge_groups_with_comprehensive_check(synonym_groups, threshold=0.96)

    # Step 1: Initialize vocabulary and calculate IDF
    vocabulary = initialize_vocabulary(vocabulary)
    vocabulary = calculate_and_update_idf(records, vocabulary)

    # Step 2: Map tokens in chunks
    for record in records:
        record['tokens'] = map_tokens_to_vocabulary(record['tokens'], vocabulary)

    # Step 3: Create BM25 vectors
    chunk_vectors = create_chunk_vectors(records, vocabulary)

    bm25_schema =     {
            "id": "INT NOT NULL PRIMARY KEY",
            "word": "VARCHAR(255) NOT NULL",
            "idf": "FLOAT",
            "synonyms": "LONGTEXT"
        }

    sql_con.delete_table(table_store)

    store_vocabulary(vocabulary=vocabulary,
                     sql_con=sql_con,
                     table_name=table_store,
                     json_path=json_path,
                     schema = bm25_schema)
    
    del bm25_schema
    del vocabulary
    del voc_analyzer
    
    # Upload and embed!!

    # Associate Chunk Sparse vectors with Metadata and Content 

    chunk_vectors_dict = {item['chunk_id']: item['sparse_vector'] for item in chunk_vectors}

    del chunk_vectors

    upsert_list = []
    for record in records:
        id_record = record['id']
        metadata_record = json.loads(record['metadata'])
        content_record = record['content']
        sparse_vector_record = chunk_vectors_dict.get(id_record, {"indices": [], "values": []})  # Default to empty sparse vector
        embed_dict = {'id':id_record,
                      'content':content_record,
                      'sparse_vector':sparse_vector_record,
                      'metadata':metadata_record}
        upsert_list.append(embed_dict)

    del chunk_vectors_dict
    del records

    return upsert_list


def create_chunk_vectors(chunks, vocabulary, k1=1.5, b=0.75):
    """
    Creates BM25 vectors for each chunk based on term frequencies and IDF values.

    Args:
        chunks (list): List of dictionaries with tokens for each chunk.
        vocabulary (dict): Structured vocabulary with `id` and `idf` values.
        k1 (float): BM25 parameter to control term frequency saturation.
        b (float): BM25 parameter to adjust for document length normalization.

    Returns:
        list: List of dictionaries containing chunk ID and BM25 vector in the required structure.
    """
    vectors = []
    avg_length = sum(len(chunk['tokens']) for chunk in chunks) / len(chunks)  # Average chunk length

    for chunk in chunks:
        chunk_id = chunk['id']  # Unique identifier for the chunk
        tokens = chunk['tokens']  # Tokens in the chunk
        length = len(tokens)  # Length of the chunk
        tf = defaultdict(int)
        
        # Calculate term frequency (TF) for the chunk
        for token in tokens:
            tf[token] += 1
        
        # Compute BM25 vector
        indices = []
        values = []
        for term, data in vocabulary.items():
            if term in tf:
                word_id = data['id']  # Use the word ID
                idf = data['idf']     # Get IDF for the term
                tf_term = tf[term]
                # BM25 weight calculation
                weight = idf * ((tf_term * (k1 + 1)) / (tf_term + k1 * (1 - b + b * (length / avg_length))))
                indices.append(word_id)  # Add word ID to indices
                values.append(weight)    # Add calculated weight to values
        
        # Store vector for the chunk
        vector_data = {
            "chunk_id": chunk_id,
            "sparse_vector": {
                "indices": indices,
                "values": values
            }
        }
        vectors.append(vector_data)

    return vectors

def initialize_vocabulary(vocabulary):
    """
    Converts a simple vocabulary with lists of synonyms into a consistent structured format.
    Assigns a unique ID to each word.

    Args:
        vocabulary (dict): Dictionary with keys as words and values as synonym lists.

    Returns:
        dict: Dictionary with a consistent structure for each entry, including IDs.
    """
    structured_vocab = {}
    for index, (word, synonyms) in enumerate(vocabulary.items()):
        structured_vocab[word] = {
            "id": index,          # Assign a unique ID based on the index
            "synonyms": synonyms, # Keep the list of synonyms
            "idf": None           # Placeholder for IDF values
        }
    return structured_vocab

def calculate_and_update_idf(records, vocabulary):
    """
    Calculates IDF values for vocabulary terms based on tokenized chunks and updates the vocabulary.
    
    Args:
        records (list): List of records with tokens for each chunk.
        vocabulary (dict): Dictionary of vocabulary words and their synonyms.
        
    Returns:
        dict: Updated vocabulary with IDF values populated.
    """
    import math
    from collections import defaultdict

    N = len(records)  # Total number of chunks
    df = defaultdict(int)  # Document frequency for each vocabulary term

    # Step 1: Map tokens in each chunk to vocabulary and calculate DF
    for record in records:
        tokens = record['tokens']  # Raw tokens from the chunk
        mapped_tokens = set(map_tokens_to_vocabulary(tokens, vocabulary))  # Map tokens to vocabulary and ensure uniqueness
        for token in mapped_tokens:
            df[token] += 1  # Increment document frequency

    # Step 2: Update vocabulary with IDF values
    for term in vocabulary:
        vocabulary[term]["idf"] = math.log(N / (df[term] + 1))  # Add 1 to DF to avoid division by zero

    return vocabulary

def map_tokens_to_vocabulary(tokens, vocabulary):
    """
    Maps tokens in a chunk to their corresponding vocabulary keys using structured synonyms.
    Args:
        tokens (list): List of tokens in a chunk.
        vocabulary (dict): Structured vocabulary with `synonyms` and `idf`.
    Returns:
        list: List of mapped tokens based on the vocabulary.
    """
    # Create a reverse mapping of synonyms to their vocabulary word
    synonym_map = {
        synonym: word
        for word, data in vocabulary.items()
        for synonym in data["synonyms"]
    }
    # Map tokens to their vocabulary word
    return [synonym_map.get(token, token) for token in tokens]

def clean_extracted_values_to_tokens(extract_values):
    """
    Function to transform and clean extracted values and words and convert them into tokens

    params:
    - extract_values (dict): Dictionary with different type of extracted values 
    
    """
    tokens_total = []
    for key,vals in extract_values.items():
        if vals:
            if key == 'parentheses_terms':
                for term in vals:
                    clean_text,extracted_terms = post_clean_document(term)
                    clean_text_tokens = extract_tokens_lemm_stop(clean_text)
                    tokens_total.extend(clean_text_tokens)
                    for key_1,val_1 in extracted_terms.items():
                        if val_1:
                            if key_1 == 'parentheses_terms':
                                for term in val_1:
                                    clean_text_1,extracted_terms_1 = post_clean_document(term)
                                    clean_text_1_tokens = extract_tokens_lemm_stop(clean_text_1)
                                    tokens_total.extend(clean_text_1_tokens)
                                    for key_2,val_2 in extracted_terms_1.items():
                                        if val_2:
                                            tokens_total.extend(val_2)
                                            print(f'Wow {key_2}')
                            else:
                                tokens_total.extend(val_1)
            else:
                tokens_total.extend(vals)

    return tokens_total

def extract_tokens_lemm_stop(text):
    
    # Create spaCy doc
    text = re.sub('\n',' ',text)
    doc = nlp(text)
    char_set = re.escape('.,-')
    # Lemmatization and stop word removal
    tokens = [
        token.lemma_.lower() for token in doc 
        if token.lemma_.lower() not in german_stopwords
        and token.lemma_.lower() not in domain_stopwords
        and not token.is_punct 
        and len(token.lemma_) > 1
        and not token.lemma_.isdigit()
    ]

    # Additional filtering: Remove tokens with only special characters or short numeric-like patterns
    tokens = [
        re.sub(rf'^[{char_set}]|[{char_set}]$', '', token) for token in tokens
        if not re.fullmatch(r'\d{1}|\d+[,\.]?\d*', token)  # Exclude short or formatted numbers
        and not re.fullmatch(r'[^\w]+', token)  # Exclude tokens with only special characters
        and len(re.sub(rf'^[{char_set}]|[{char_set}]$', '', token)) > 1
    ]

    
    return tokens


class WordFrequencyAnalyzer:
    def __init__(self):
        # Initialize an empty Counter to store word frequencies
        self.word_frequencies = Counter()

    def update_words(self, words):
        """
        Updates the word frequency counter with a new list of words.
        
        Args:
            words (list): List of words to add/update in the frequency count.
        """
        self.word_frequencies.update(words)

    def get_top_words(self, n=10):
        """
        Returns the top n most frequent words, ordered by frequency.
        
        Args:
            n (int): Number of top frequent words to return.
        
        Returns:
            list: List of tuples (word, frequency).
        """
        return self.word_frequencies.most_common(n)

    def word_character_count(self):
        """
        Returns a dictionary with words as keys and their character counts as values.
        
        Returns:
            dict: Dictionary with word character counts.
        """
        return {word: len(word) for word in self.word_frequencies.keys()}
    
def normalize_umlauts(word):
    """
    Normalizes umlauts and special German characters to equivalent ASCII forms.
    Removes hyphens as well.
    """
    return word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss").replace("-", "")

def all_start_with_same_prefix(words, prefix_length=2):
    """
    Checks if all words in the list start with the same prefix after normalization.
    """
    if not words:
        return False

    # Normalize words
    normalized_words = [normalize_umlauts(word) for word in words]

    # Get the prefix of the first word
    prefix = normalized_words[0][:prefix_length]

    # Check if all normalized words start with the same prefix
    return all(word.startswith(prefix) for word in normalized_words)

def group_synonyms_with_prefix_check(
    vocabulary, 
    threshold=0.95, 
    exclude_words=None, 
    ignore_norms_pattern=None, 
    prefix_length=2
):
    """
    Groups similar words from a vocabulary while handling exclusions and specific patterns.

    Args:
        vocabulary (list of str): The list of words to process.
        threshold (float): The similarity threshold (0-1) for grouping words.
        exclude_words (set): Words or prefixes to check for exclusion.
        ignore_norms_pattern (str): Regex pattern for norms to exclude.
        prefix_length (int): Minimum prefix length for words in a group to be considered a match.

    Returns:
        dict: A dictionary where keys are representative words, and values are lists of their synonyms.
    """
    # Normalize patterns directly into their own groups
    nummer_words = [word for word in vocabulary if word.startswith("Nummer")]
    buchstabe_words = [word for word in vocabulary if word.startswith("Buchstabe")]
    other_words = [word for word in vocabulary if word not in nummer_words + buchstabe_words]

    groups = {}  # Dictionary to store groups of synonyms

    # Add Nummer and Buchstabe words directly as separate groups
    for word in nummer_words:
        groups[word] = [word]
    for word in buchstabe_words:
        groups[word] = [word]

    processed = set()  # Set to keep track of processed words
    if exclude_words is None:
        exclude_words = set()

    for word in other_words:
        # Skip if already processed
        if word in processed:
            continue

        # Normalize the word
        normalized_word = normalize_umlauts(word)

        # Check for exclusions
        if ignore_norms_pattern and re.search(ignore_norms_pattern, word):
            continue

        # Initialize the current word as a potential new group
        current_group = [word]
        processed.add(word)
        representative = word  # Initially, the current word is the representative

        # Check against existing groups to merge
        to_merge = []  # List of groups that overlap with the current word
        for rep, group in groups.items():
            # Check if the word matches any word in the current group
            if any(
                Levenshtein.ratio(normalized_word, normalize_umlauts(g)) >= threshold
                for g in group if g not in exclude_words
            ):
                # Ensure all words in the group have the same prefix
                if not all_start_with_same_prefix(
                    [w for w in group + [word] if w not in exclude_words], prefix_length
                ):
                    continue
                to_merge.append(rep)

        # Merge all overlapping groups into the current group
        for rep in to_merge:
            current_group.extend(groups.pop(rep))

        # Check all unprocessed words
        for other_word in other_words:
            if other_word in processed or other_word == word:
                continue

            # Normalize umlauts for comparison
            normalized_other_word = normalize_umlauts(other_word)

            # Calculate similarity
            similarity = Levenshtein.ratio(normalized_word, normalized_other_word)
            if similarity >= threshold:
                # Ensure all words in the group have the same prefix
                if not all_start_with_same_prefix(
                    [w for w in current_group + [other_word] if w not in exclude_words], prefix_length
                ):
                    continue
                current_group.append(other_word)
                processed.add(other_word)

        # Update the representative to the smallest word in the group
        representative = min(current_group, key=len)
        groups[representative] = list(set(current_group))  # Remove duplicates

    return groups

def merge_groups_with_comprehensive_check(
    groups, 
    threshold=0.95
):
    """
    Merges groups by comparing each group's main word with all synonyms in subsequent groups.
    Uses the min and max of normalized synonyms as representatives for more accurate comparison.
    Skips words starting with 'Nummer' or 'Buchstabe'.

    Args:
        groups (dict): Dictionary where keys are main words, and values are lists of synonyms.
        threshold (float): Similarity threshold for merging.

    Returns:
        dict: Updated dictionary of merged groups.
    """
    def normalize_umlauts_hyphens(word):
        """Normalizes umlauts and removes hyphens."""
        return word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss").replace("-", "")

    # Separate groups for Nummer and Buchstabe words
    nummer_groups = {rep: synonyms for rep, synonyms in groups.items() if rep.startswith("Nummer")}
    buchstabe_groups = {rep: synonyms for rep, synonyms in groups.items() if rep.startswith("Buchstabe")}

    # Filter out Nummer and Buchstabe groups from the main processing
    main_groups = {rep: synonyms for rep, synonyms in groups.items() if rep not in nummer_groups and rep not in buchstabe_groups}

    representatives = list(main_groups.keys())  # List of all representative terms
    merged_groups = {}  # Dictionary to hold merged groups
    processed = set()  # Keep track of already processed groups

    for i, rep1 in enumerate(representatives):
        # Skip if this representative is already processed
        if rep1 in processed:
            continue

        # Normalize the synonyms of the current group
        current_group = main_groups[rep1]
        normalized_group = [normalize_umlauts_hyphens(word) for word in current_group]

        # Calculate representatives as min and max if there are multiple synonyms
        if len(current_group) > 1:
            current_representatives = [min(normalized_group), max(normalized_group)]
        else:
            current_representatives = [normalize_umlauts_hyphens(rep1)]

        # Start a new group for this representative
        merged_groups[rep1] = set(main_groups[rep1])
        processed.add(rep1)

        # Compare rep1's representatives with all subsequent groups
        for rep2 in representatives[i + 1:]:
            # Skip if this representative is already processed
            if rep2 in processed:
                continue

            # Normalize the synonyms of the comparison group
            comparison_group = main_groups[rep2]
            normalized_comparison_group = [normalize_umlauts_hyphens(word) for word in comparison_group]

            # Skip Nummer and Buchstabe words
            if rep2.startswith("Nummer") or rep2.startswith("Buchstabe"):
                continue

            # Check if any of the current representatives matches any word in the comparison group
            if any(
                Levenshtein.ratio(rep, comp) >= threshold
                for rep in current_representatives
                for comp in normalized_comparison_group
            ):
                # Merge the groups
                merged_groups[rep1].update(main_groups[rep2])
                processed.add(rep2)  # Mark rep2 as processed

    # Convert sets back to lists
    merged_groups = {rep: list(synonyms) for rep, synonyms in merged_groups.items()}

    # Add Nummer and Buchstabe groups back into the final result
    merged_groups.update(nummer_groups)
    merged_groups.update(buchstabe_groups)

    return merged_groups


def process_and_upload(
    chunks,
    pinecone_connector,
    embedding_handler,
    use_sparse=True,
    batch_size_embedding=100,
    batch_size_upsert=25
):
    """
    Processes document chunks to generate embeddings and uploads them to Pinecone with a batch-level progress bar.

    Args:
        chunks (list): List of chunks, each containing 'id', 'content', 'metadata', and 'sparse_vector'.
        pinecone_connector (PineconeDBConnectorHybrid): Pinecone connector instance.
        embedding_handler (EmbeddingHandler): Embedding handler for generating dense embeddings.
        use_sparse (bool): Whether to include sparse vectors.
        batch_size_embedding (int): Number of chunks to process in each embedding batch.
        batch_size_upsert (int): Number of vectors to upload to Pinecone in each upsert batch.
    """
    total_batches = (len(chunks) + batch_size_embedding - 1) // batch_size_embedding  # Total number of batches
    progress_bar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")

    for i in range(0, len(chunks), batch_size_embedding):
        batch = chunks[i:i + batch_size_embedding]
        ids = [chunk['id'] for chunk in batch]
        texts = [chunk['content'] for chunk in batch]
        metadatas = [chunk['metadata'] for chunk in batch]

        # Step 1: Generate Dense Embeddings
        dense_embeddings = embedding_handler.embed_texts(texts)

        # Step 2: Prepare Data for Upload
        vectors_to_upload = []
        for j, embedding in enumerate(dense_embeddings):
            vector = {
                "id": ids[j],
                "values": embedding,
                "metadata": metadatas[j],
            }
            if use_sparse:
                sparse_vector = batch[j].get('sparse_vector', {"indices": [], "values": []})
                if sparse_vector['indices'] and sparse_vector['values']:  # Check if sparse vector exists
                    vector["sparse_values"] = {
                        "indices": sparse_vector["indices"],
                        "values": sparse_vector["values"]
                    }
            vectors_to_upload.append(vector)

        # Step 3: Upload to Pinecone
        for j in range(0, len(vectors_to_upload), batch_size_upsert):
            upsert_batch = vectors_to_upload[j:j + batch_size_upsert]
            pinecone_connector.upsert_vectors(vectors=upsert_batch)

        # Update the batch progress bar
        progress_bar.update(1)

    progress_bar.close()
    print("All batches uploaded successfully.")