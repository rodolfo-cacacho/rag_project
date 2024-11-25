import os
import json
import pandas as pd
import numpy as np
import re
import ollama
import openai
from dotenv import load_dotenv
from database_manager import MySQLDB
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document


load_dotenv()
API_KEY_CGPT = os.getenv('API_KEY_CGPT')
print(f'Loaded API KEY ChatGPT: {API_KEY_CGPT}')
openai.api_key = API_KEY_CGPT
MODEL="gpt-4o"

# Config and initialization
config = {
    'user': 'root',
    'password': 'admin123',
    'host': '127.0.0.1'
}

db_name = 'data_rag'

table_documents_name = 'table_documents'

schema_table_documents = {
  'id': 'int NOT NULL AUTO_INCREMENT PRIMARY KEY',
  'pdf_type': 'varchar(90) NOT NULL',
  'pdf_name': 'varchar(90) NOT NULL',
  'content': 'longtext NOT NULL',
  'page': 'varchar(90) NOT NULL',
  'type': 'varchar(90) NOT NULL',
  'metadata': 'longtext NOT NULL'
}

def init_db_connection(db_name=db_name,config=config,table_name_documents = table_documents_name, schema_documents = schema_table_documents):

    db = MySQLDB(config, db_name)
    db.create_table(table_name_documents, schema_documents)

    return db


default_conversation = [
    {"role": "system",
     "content": '''You are a text classifier. Explain why you chose that classification.'''}
]

def add_question(conversation,question):
    chat = {"role": "user", "content": question}
    conversation_t = conversation.copy()
    conversation_t.append(chat)
    return conversation_t

def count_words(text):
    """
    Counts the number of words in a given text.

    Parameters:
    text (str): The text to count words in.

    Returns:
    int: The number of words in the text.
    """
    # Split the text by whitespace and count the resulting list items
    words = text.split()
    return len(words)


def get_response(text,conversation,model = 'llama'):
    question = f'''You have to classify between these 2 categories: 'Thema/Stichwort' or 'Beschreibung'. The text is in german. 'Thema/Stichwort' are key words, topic, concepts or phrases and usually don't have verbs. 'Beschreibung' is a longer text describing details, given in a sentence structure and uses verbs.\nThe text is the following: {text}'''
    conversation_t = add_question(conversation,question)
    # print(conversation_t)
    if model == 'llama':
        response = ollama.chat(model='llama3', messages=conversation_t)
        
        answer = response['message']['content']
        print(f'Text {text[:100]} Answer: {answer}')
        
        return answer
    
    elif model == 'openai':
            response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Change this to the desired model (e.g., "davinci" or "curie")
            messages=conversation_t,
            max_tokens=100  # Adjust the maximum number of tokens for the response
        )
            answer = response.choices[0].message.content
            print(f'Text {text[:100]} Answer: {answer}')
            return answer
    
    else:
        count = count_words(text) 
        print(f'Words: {count} - Text: {text}')
        return count

def clean_spaces(text):
    if isinstance(text,str):
        cleaned_txt = re.sub(r'\s+',' ',text)
        cleaned_txt = cleaned_txt.strip()
        return cleaned_txt
    else:
        return text
    
def clean_spaces_list(list):
    return [clean_spaces(text) for text in list]

def categorize_text(text,conversation,model):
    """
    Categorizes the given text into one of the following categories:
    'Nr.', 'Thema/Stichwort', 'Beschreibung', 'Relevanz', or 'empty'.

    Parameters:
    text (str): The input text to categorize.

    Returns:
    str: The category of the text.
    """
    if not text or text.strip() == '':
        return 'empty'
    
    # Check if text is a number (Nr.)
    if re.match(r'^\d+\.\d{2}$', text):
        return 'Nr.'
    
    # Check if text is a relevance value (WG, NWG, WG, NWG)
    if text.strip() in {'WG', 'NWG', 'WG, NWG'}:
        return 'Relevanz'
    
    if count_words(text) <11:
        return 'Thema/Stichwort'
    
    else:
        return 'Beschreibung'

    # else:
    #     return classify_text(text,conversation,model)

def classify_text(text,conversation,model):

    answer = get_response(text,conversation,model=model)
    return answer

def merge_tables(csv_list):
    for index,file in enumerate(csv_list):
        print(file)
        df = pd.read_csv(file)
        # print(df)
        df.fillna('',inplace=True)
        # print(df.head(5))

        columns = clean_spaces_list(list(df))
        print(f'Header: {columns}')
        cleaned_rows = []
        pattern_question = r'^(\d+\.\d{2}) (.+)$'
        pattern_question2 = r'(\d+\.\d+\s*-\s*\d+\.\d+)\s+(.+)'

        for i,row in df.iterrows():
            row_list = clean_spaces_list(row.tolist())
            
            # print(row_list)
            for j,item in enumerate(row_list):
                if item in columns:
                    row_list[j] = ''
                matches = re.match(pattern_question,item)
                matches2 = re.match(pattern_question2,item)
                if bool(matches) and not bool(matches2):
                    q_no = matches.group(1)
                    text_no = matches.group(2)
                    # if(q_no.startswith('4')):
                    #     print(f'{q_no}\n{text_no}')
                    row_list[j]=q_no
                    if(row_list[j+1]== ''):
                        row_list[j+1]=text_no
                    elif(row_list[j+2]==''):
                        row_list[j+2]=row_list[j+1]
                        row_list[j+1]=text_no
                    elif(row_list[j+3]==''):
                        row_list[j+3]=row_list[j+2]
                        row_list[j+2]=row_list[j+1]
                        row_list[j+1]=text_no
                elif bool(matches2):
                    matches2 = re.match(pattern_question2,item)
                    q_no = matches2.group(1)
                    text_no = matches2.group(2)
                    print(f'{q_no} \n {text_no}')
                    row_list[j]=q_no
                    if(row_list[j+1]== ''):
                        row_list[j+1]=text_no
                    elif(row_list[j+2]==''):
                        row_list[j+2]=row_list[j+1]
                        row_list[j+1]=text_no
                    elif(row_list[j+3]==''):
                        row_list[j+3]=row_list[j+2]
                        row_list[j+2]=row_list[j+1]
                        row_list[j+1]=text_no
                
                
                    # print(f'item: {item} row {row_list}')
            if (any(item != '' for item in row_list)):
                cleaned_rows.append(row_list)

        if index > 0:
            append_df = pd.DataFrame(cleaned_rows,columns=columns)
            result_df = pd.concat([result_df, append_df], ignore_index=True)

        else:
            result_df = pd.DataFrame(cleaned_rows,columns=columns)
        
        print(f'DF shape {result_df.shape}')

    return result_df

def find_last_index(lst, text):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == text:
            return i
    return -1

def clean_data_frame(df):

    columns = clean_spaces_list(list(df))
    print(f'Header: {columns}')
    cleaned_rows = []
    pattern_question = r'^(\d+\.\d{2}) (.+)$'

    for i,row in df.iterrows():
        row_list = clean_spaces_list(row.tolist())
        
        # print(row_list)
        for j,item in enumerate(row_list):
            if item in columns:
                row_list[j] = ''
            matches = re.match(pattern_question,item)
            if bool(matches):
                q_no = matches.group(1)
                text_no = matches.group(2)
                # print(f'{q_no}\n{text_no}')
                row_list[j]=q_no
                if(row_list[j+1]== ''):
                    row_list[j+1]=text_no
            
                # print(f'item: {item} row {row_list}')
        if (any(item != '' for item in row_list)):
            cleaned_rows.append(row_list)
        
    cleaned_df = pd.DataFrame(cleaned_rows,columns=columns)

    # previous_row_cat = []
    # previous_row = []
    act_row_cat = []
    row_list = []
    elements = []
    elements_cat = []
    for i,row in cleaned_df.iterrows():
        # previous_row_cat = act_row_cat.copy()
        # previous_row = row_list.copy()
        row_list = row.tolist()
        act_row_cat = []
        for j,item in enumerate(row_list):
            elements.append(item)
            type = categorize_text(item,default_conversation,model = 'count')
            elements_cat.append(type)
            act_row_cat.append(type)
        # if act_row_cat not in possible_rows:
            # print(f'cat prev: {previous_row_cat} cat act: {act_row_cat}\ntext prev: {previous_row[:20]}\ntext act: {row_list[:20]}\n\n')

    return elements,elements_cat

def clean_table_order(cat_list,content_list):
    allowed_patterns = [['Nr.','Thema/Stichwort'],['Thema/Stichwort','Nr.'],['Thema/Stichwort','Beschreibung'],['Beschreibung','Relevanz'],['Relevanz','Nr.'],['Beschreibung','Nr.']]
    cleaned_key_list = []
    cleaned_target_list = []

    for key, target in zip(cat_list, content_list):
        if key != 'empty':
            cleaned_key_list.append(key)
            cleaned_target_list.append(target)
    fix = 0
    l_list = len(cleaned_key_list)
    new_clean_list_cat = []
    new_clean_list = []
    for index,key in enumerate(cleaned_key_list):
        text = cleaned_target_list[index]
        if fix == 0 and index < l_list-1:
            pair = [key,cleaned_key_list[index+1]]
            # print(pair)
            new_clean_list_cat.append(key)
            new_clean_list.append(text)
            if pair not in allowed_patterns:
                # print('error')
                fix = 1
        elif index == l_list-1:
                if key == 'Beschreibung':
                    target_index = find_last_index(new_clean_list_cat,'Beschreibung')
                    new_clean_list_cat[target_index] = new_clean_list_cat[target_index]
                    new_clean_list[target_index] = new_clean_list[target_index] + ' ' + text
                else:
                    new_clean_list_cat.append(key)
                    new_clean_list.append(text)
        else: # Skip
            if key == 'Beschreibung':
                # Append to previous text
                target_index = find_last_index(new_clean_list_cat,'Beschreibung')
                new_clean_list_cat[target_index] = new_clean_list_cat[target_index]
                new_clean_list[target_index] = new_clean_list[target_index] + ' ' + text
                # new_clean_list.append('fix')
            fix = 0
    
    data = {
        'Nr.':[],
        'Thema/Stichwort':[],
        'Beschreibung':[],
        'Relevanz':[]
    }
    for index,item in enumerate(new_clean_list_cat):
        if index < len(new_clean_list_cat)-1:
            next_item = new_clean_list_cat[index+1]
            text = new_clean_list[index]
            if next_item == 'Nr.':
                data[item].append(text)
                cant = len(data[next_item])
                for key, value in data.items():
                    if len(value) < cant:
                        value.append('')
            else:
                data[item].append(text)
        else:
            text = new_clean_list[index]
            data[item].append(text)

    # Iterating over the dictionary
    for key, value in data.items():
        print(f"{key}:{len(value)}")
    data_frame_final = pd.DataFrame(data)

    return data_frame_final

# Function to extract the number from the filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def clean_csv_file(csv_directory):

    files = os.listdir(csv_directory)
    filtered_files = [file for file in files if file.endswith('.csv')]
    filtered_files = [file for file in filtered_files if extract_number(file) != 0]
    filtered_files = sorted(filtered_files, key=extract_number)
    full_paths = [os.path.join(csv_directory, filename) for filename in filtered_files]
    
    result_df = merge_tables(full_paths)
    content,category = clean_data_frame(result_df)

    # for i,c in enumerate(content):
    #     if(category[i] != 'empty'):
    #         print(f'{i} - {category[i]} : {content[i]}\n')

    df_final = clean_table_order(category,content)

    return df_final

def add_main_topic_column(df, number_col='Nr.', topic_col='Thema/Stichwort'):
    # Sort the DataFrame by the number column in ascending order
    df = df.sort_values(by=number_col).reset_index(drop=True)
    
    # Add a new column for main topics
    df['Hauptthema'] = None
    
    # Extract the main topic for each group
    for group, group_df in df.groupby(df[number_col].astype(str).str.split('.').str[0]):
        # Get the main topic from the first row in the group
        main_topic = group_df.iloc[0][topic_col]
        
        # Assign the main topic to the rest of the group
        df.loc[group_df.index, 'Hauptthema'] = main_topic
        
        # Remove the row used to determine the main topic
        df = df.drop(group_df.index[0])
    
    # Reorder columns: number, main_topic, topic, description, relevance
    cols = list(df.columns)
    main_topic_index = cols.index('Hauptthema')
    # Move 'main_topic' to the second position
    cols.insert(1, cols.pop(main_topic_index))
    df = df[cols]
    
    return df


"""
    total_elements = get_content_table(total_elements,db_con,pdf_file_name,pdf_file_type,db_table_name,directory)
    total_elements = id_and_hierarchy(total_elements)
    total_elements = group_pages_elements(total_elements)
    documents = metadata_update(total_elements,version_file,pdf_file_name,pdf_file_type,directory,file_image_storage,tables_folder_name)
    store_documents(documents,db_con,db_doc_name,pdf_file_name,pdf_file_type)

    TO DO:
    1. Clean document/csv
    2. Hierarchy / Important since this are questions
    3. More important to group by question and not page
    4. Update Metadata to look like the one in the other files
    5. Store documents in the database

"""

def clean_faq_beg_csv(directory,output_dir):

    dirs = os.listdir(directory)
    dirs = [item for item in dirs if os.path.isdir(os.path.join(directory, item)) and item != 'processed_tables']
    base_folder_name = os.path.basename(directory.rstrip('/\\'))
    # Create the base folder in the target_directory
    target_base_folder = os.path.join(output_dir, base_folder_name)
    os.makedirs(target_base_folder, exist_ok=True)

    for i,dir in enumerate(dirs):
        print(f'Working on {dir} - {i+1}/{len(dirs)}...\n')
        dir_tables = os.path.join(directory,dir,'tables')
        out_dir = os.path.join(target_base_folder,dir)
        os.makedirs(out_dir, exist_ok=True)
        output_name = os.path.join(out_dir,(dir+'.csv'))
        print(output_name)
        df_processed = clean_csv_file(dir_tables)
        df_processed = add_main_topic_column(df_processed)
        df_processed.to_csv(output_name, sep=',', index=False, header=True)

    print("Done........")

def clean_csv(csv_directory,processed_csv_name = 'clean_table.csv'):

    # Process if exist
    if os.path.exists(csv_directory):
        print(f'CSV Path Exists: {csv_directory}')

        target_location = os.path.dirname(csv_directory)

        output_name = os.path.join(target_location,processed_csv_name)

        if not os.path.exists(output_name):
            df_processed = clean_csv_file(csv_directory)
            df_processed = add_main_topic_column(df_processed)
            df_processed.to_csv(output_name, sep=',', index=False, header=True)
        else:
            print(f'{output_name} already processed')

        return output_name
    
    else:
        return 'Not Found'
    
def process_pdf(pdf_file,pages = 1):

    if os.path.exists(pdf_file):
        loader_pdf = PyMuPDFLoader(pdf_file,extract_images=False)
        data_pdf = loader_pdf.load()
        data_pdf = data_pdf[:pages]
        
        return data_pdf
    
    else: 
    
        return []

def process_csv(csv_file):

    if(os.path.exists(csv_file)):
        print('CSV exists\n')
        loader_csv = CSVLoader(file_path=csv_file)
        data_csv = loader_csv.load()

        return data_csv
    else:
        return []
    


def update_metadata(elements,pdf_type,pdf_name,version_number,publication_date):

    pattern = r"Nr\.\s*:\s*\d+\.\d+"

    for i,element in enumerate(elements):

        if 'page' in element.metadata:
            page_element = element.metadata['page']
        else:
            row = element.page_content
            match = re.search(pattern, row)

            # Extract the matched pattern
            if match:
                page_element = match.group(0)
            else:
                page_element = ''
        type_element = 'Text'

        met_doc = {'doc_type':pdf_type,
            'page': page_element,
            'type':type_element,
            'source':pdf_name,
            'path':"",
            'version':version_number,
            'valid_date':publication_date}
        
        element.metadata = met_doc

    return elements
    
def store_documents(documents,db_connection,table_name,pdf_file_name,pdf_file_type):

    doc_list = []
    for i in documents:
        page_i = i.metadata['page']
        type_i = i.metadata['type']
        list_element = {'pdf_type':pdf_file_type,
                        'pdf_name':pdf_file_name,
                        'content':i.page_content,
                        'page':page_i,
                        'type':type_i,
                        'metadata':json.dumps(i.metadata)}
        doc_list.append(list_element)
    db_connection.insert_many_records(table_name,doc_list)

def process_directory_csv(original_file_directory,metadata_directory='data/documents/metadata'):
    
    output_directory = original_file_directory.replace('original','output')
    pdfs = os.listdir(original_file_directory)
    pdfs = [os.path.join(original_file_directory,f) for f in pdfs if f.endswith('.pdf')]
    version_file = os.path.join(metadata_directory,'Files_date_version.csv')

    csv_version = pd.read_csv(version_file)

    
    docs = []
    for pdf in pdfs:

        name_pdf = os.path.basename(pdf)
        matching_row = csv_version.loc[csv_version['file'] == name_pdf]
        if not matching_row.empty:
            version_number = matching_row['version'].iloc[0]
            publication_date = matching_row['date'].iloc[0]
        name = os.path.splitext(name_pdf)
        type_pdf = os.path.basename(os.path.dirname(pdf))
        dict = {'name':name,
                'pdf_path':pdf,
                'pdf_name': name_pdf,
                'type': type_pdf,
                'version' : version_number,
                'publication_date' : publication_date}
        docs.append(dict)
    
    csv_file_roots = os.listdir(output_directory)
    
    for i,folder in enumerate(csv_file_roots):

        csv_folder = os.path.join(output_directory,folder,'tables')

        if os.path.exists(csv_folder):

            csv_clean_file = clean_csv(csv_folder)
            for j in docs:
                if folder in j['name']:
                    j['csv'] = csv_clean_file
    documents = []
    for i in docs:
        print(i)
        elements = []
        elements.extend(process_pdf(i['pdf_path']))
        elements.extend(process_csv(i['csv']))

        elements = update_metadata(elements = elements,
                                   pdf_type=i['type'],
                                   pdf_name=i['pdf_name'],
                                   version_number=i['version'],
                                   publication_date=i['publication_date'])

        documents.append(elements)

    store_documents(documents)

    return documents

    

