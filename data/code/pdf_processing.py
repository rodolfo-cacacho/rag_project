import os
import json
import re
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import base64
from dotenv import load_dotenv
import openai
import pandas as pd
import shutil
from database_manager import MySQLDB
import unicodedata



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
table_description_name = 'table_description'
table_documents_name = 'table_documents'
schema_table_description = {
  'id': 'int NOT NULL AUTO_INCREMENT PRIMARY KEY',
  'pdf_type': 'varchar(90) NOT NULL',
  'pdf_name': 'varchar(90) NOT NULL',
  'table_path': 'varchar(90) NOT NULL',
  'description': 'longtext NOT NULL',
  'footnote': 'longtext'
}
schema_table_documents = {
  'id': 'int NOT NULL AUTO_INCREMENT PRIMARY KEY',
  'pdf_type': 'varchar(90) NOT NULL',
  'pdf_name': 'varchar(90) NOT NULL',
  'content': 'longtext NOT NULL',
  'page': 'varchar(90) NOT NULL',
  'type': 'varchar(90) NOT NULL',
  'metadata': 'longtext NOT NULL'
}

def init_db_connection(db_name=db_name,table_name_desc_tables=table_description_name,schema_tables=schema_table_description,config=config,table_name_documents = table_documents_name, schema_documents = schema_table_documents):

    db = MySQLDB(config, db_name)
    db.create_table(table_name_desc_tables, schema_tables)
    db.delete_table(table_documents_name)
    db.create_table(table_name_documents, schema_documents)

    return db


"""

Complete processing (PDF + CSV)

"""

def process_all_docs(folder,csv_type,img_store,max_pages=50):
    total_docs = []
    folders = os.listdir(folder)
    folders_filt = [i for i in folders if i not in csv_type and os.path.isdir(os.path.join(folder,i))]
    folders_csv = [i for i in folders if i in csv_type and os.path.isdir(os.path.join(folder,i))]
    n = 0

    db_connection = init_db_connection()
    print("DB Connection done...")

    for i in folders_filt:
        print(f'processing {i}... {(n+1)}/{len(folders_filt)+len(folders_csv)}')
        path_i = os.path.join(folder,i)
        folders_i = os.listdir(path_i)
        folders_i_filt = [t for t in folders_i if os.path.isdir(os.path.join(folder,i,t))]
        k = 0
        for j in folders_i_filt:
            print(f'processing sub_folder {j} - {k+1}/{len(folders_i_filt)}\n')
            j_path = os.path.join(folder,i,j)
            docs = process_directory_pdf(j_path,img_store,db_connection, max_pages=max_pages)
            total_docs.append(docs)
            k+=1
        n+=1
    for i in folders_csv:
        print(f'processing {i}... {(n+1)}/{len(folders_csv)+len(folders_filt)}')
        path_i = os.path.join(folder,i)
        folders_i = os.listdir(path_i)
        folders_i_csv = [t for t in folders_i if os.path.isdir(os.path.join(folder,i,t))]
        k = 0
        for j in folders_i_csv:
            print(f'processing sub_folder {j} - {k+1}/{len(folders_i_csv)}\n')
            j_path = os.path.join(folder,i,j)
            docs = process_directory_csv(output_directory=j_path,db_connection=db_connection)
            total_docs.append(docs)
            k+=1
        n+=1

    return total_docs


"""

PDF Processing

"""


def process_directory_pdf(directory,file_image_storage, db_connection,max_pages=50,default_json_file = 'structuredData.json',default_tables_folder = 'tables',metadata_directory = 'data/documents/metadata',db_table_name = table_description_name,db_doc_name = table_documents_name):
    # Identify part folders and sort them
    
    pdf_file_name = os.path.basename(directory)+'.pdf'
    pdf_file_type = os.path.basename(os.path.dirname(directory))
    # print(f'pdf file: {pdf_file_name} - pdf type: {pdf_file_type}')

    part_folders = sorted([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and f.startswith('part')])

    # Initialize offset
    offset_page = 0
    json_file_name = default_json_file
    tables_folder_name = default_tables_folder

    total_elements = []
    offset_apply = False
    if part_folders:
        offset_apply = True
        for part in part_folders:
            # print(f'page offset {offset_page}')
            offset_page,elements = process_folder(part, offset_page, max_pages,json_file_name,offset_apply)
            total_elements.extend(elements)
    else:
        offset_page,elements = process_folder(directory, offset_page, max_pages,json_file_name,offset_apply)
        total_elements.extend(elements)

    version_file = os.path.join(metadata_directory,'Files_date_version.csv')

    db_con = db_connection

    total_elements = get_content_table(total_elements,db_con,pdf_file_name,pdf_file_type,db_table_name,directory)
    total_elements = id_and_hierarchy(total_elements)
    total_elements = group_pages_elements(total_elements)
    documents = metadata_update(total_elements,version_file,pdf_file_name,pdf_file_type,directory,file_image_storage,tables_folder_name)
    store_documents(documents,db_con,db_doc_name,pdf_file_name,pdf_file_type)


    return(documents)

def apply_offsets_to_elements(elements, page_offset, folder):
    # Apply offsets to page numbers and objIDs in the JSON data
    part_name = os.path.basename(folder)
    for item in elements:
        if 'page' in item:
            item['page'] += page_offset
        if 'filePaths' in item:
            item['filePaths'] = os.path.join(part_name, item['filePaths'])
    return elements

def process_folder(folder, page_offset, max_pages, json_file_name, offset_apply):
    # Load and process the JSON file
    json_file_path = os.path.join(folder, json_file_name)
    if os.path.exists(json_file_path):
        # print(f'reading file: {json_file_path}')
        data_json = read_json(json_file_path)
            # Apply offset to JSON data
            # data = apply_offset_to_json(data, offset)
        elements = process_json(data_json)
        elements = compress_labels(elements)
        elements = separate_tables(elements)
        if offset_apply:
            elements = apply_offsets_to_elements(elements, page_offset, folder)

    new_page_offset = page_offset + max_pages

    return new_page_offset, elements

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
                        'metadata':json.dumps(i.metadata,ensure_ascii=False)}
        doc_list.append(list_element)
    db_connection.insert_many_records(table_name,doc_list)

def process_json(json_data):
    # Implement your JSON processing logic here
    extracted_items = []
    current_table = None
    # print("Processing JSON data: ")
    if 'elements' in json_data:
        for element in json_data['elements']:
            path = element.get('Path', '')
            item_type = get_last_element(path)
            text = element.get('Text', '')
            text_size = element.get('TextSize', 0)
            font_name = element.get('Font', {}).get('name', '')
            page = element.get('Page', '')

            # Check if the current element is part of a table
            if 'Table' in path:
                # If it's a new table, record it
                if current_table != path:
                    current_table = path
                    if 'filePaths' in element:
                        extracted_items.append({
                            'filePaths': element['filePaths'],
                            'page':page,
                            'type': item_type
                        })
            else:
                # If it's a non-table element, reset current_table
                current_table = None

                # Store the categorized element
                extracted_items.append({
                    'content': text.strip(),
                    'textSize': text_size,
                    'font': font_name,
                    'page':page,
                    'type': item_type,
                })

    return extracted_items

def compress_labels(elements):
    possible_elements = ['LBody','P','ParagraphSpan',]
    new_elements = []
    skip = None
    for n,i in enumerate(elements):
        if skip == None:
            type_i = i['type']
            if n < len(elements) - 1:
                type_i2 = elements[n+1]['type']
                if type_i == 'Lbl' and type_i2 in possible_elements:
                    i['content'] = i['content']+' '+elements[n+1]['content']
                    skip = True
                    i['type'] = 'Item'
            new_elements.append(i)
        else:
            skip = None

    return new_elements

def separate_tables(elements):
    new_elements = []
    for n,i in enumerate(elements):

        i_type = i['type']
        if i_type == 'Table':
            valid_tables = [t for t in i['filePaths'] if t.endswith('png')]
            # print(valid_tables)
            page_n = 0
            if len(valid_tables) > 1:
                for j in valid_tables:
                    copy_i = i.copy()
                    copy_i['page'] += page_n
                    page_n += 1
                    copy_i['filePaths'] = j
                    new_elements.append(copy_i)
            else:
                copy_i = i.copy()
                copy_i['filePaths'] = valid_tables[0]
                new_elements.append(copy_i)
        else:
            new_elements.append(i)

    return new_elements

def group_pages_elements(elements):

    new_elements = []
    act_page = None
    for i, element in enumerate(elements):
        type_element = element['type']
        page_element = element['page']
        # if type_element == 'Table':
        #     element['content'] = 'Description of the table... Description of the table... Description of the table...'
        content = element['content']
        # print(f'type {type_element} - {page_element}')
        if act_page != page_element:
            if i > 0 and act_page!= 'Table':
                new_elements.append(new_page_element)
            new_page_element = element
            act_page = page_element
        else:
            if type_element != 'Table':
                new_page_element['content'] = new_page_element['content'] + '\n' + content
                act_page = page_element
                if i == len(elements)-1:
                    new_elements.append(new_page_element)
            elif type_element == 'Table':
                if i > 0:
                    new_elements.append(new_page_element)
                new_elements.append(element)
                act_page = 'Table'

    return new_elements

def id_and_hierarchy(elements):
    ## Add IDs first
    id_n = 1
    for i in elements:
        i['Id'] = id_n
        id_n += 1
    ## Group by hierarchical elements
    elements = create_hierarchy(elements)

    return elements

# Load the JSON file
def read_json(file_read):
    with open(file_read, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to extract the last element from an XPath and remove numeric indexes
def get_last_element(xpath):
    # Use regular expression to find the last element and remove numeric indexes
    match = re.search(r'([^/]+)$', xpath)
    if match:
        element = match.group(1)
        # Remove numeric indexes inside brackets
        element = re.sub(r'\[\d+\]', '', element)
        return element
    return None

def metadata_update(elements,version_file,pdf_name,pdf_type,directory,file_image_storage,tables_folder_name):

    # Goal get version and date
    ## Check if file in the table
    pdf_name_wo_ext, _ = os.path.splitext(pdf_name)


    csv_version = pd.read_csv(version_file)
    matching_row = csv_version.loc[csv_version['file'] == pdf_name]
    # Check if any match was found
    if not matching_row.empty:
        version_number = matching_row['version'].iloc[0]
        publication_date = matching_row['date'].iloc[0]
        type_key = int(matching_row['type_key'].iloc[0])
        file_key = int(matching_row['file_key'].iloc[0])
        # print(f'Version: {version_number} - Date: {publication_date}\n')
    else:
        ## extract 1st,2nd and last page:
        elements_target = []
        pages_target = [0,1]
        for i in elements:
            page_count = i['page']
            if page_count in pages_target:
                elements_target.append(i)
    
        matching_dicts = [d for d in elements if d.get('page') == page_count]
        elements_target.extend(matching_dicts)

        context = ''
        act_page = None
        for n,i in enumerate(elements_target):
            page = i['page']
            content = i['content']
            if act_page != page:
                if n == 0:
                    context += f'Page: {page}\n{content}'
                else:
                    context += f'\nPage: {page}\n{content}'
                act_page = page
            else:
                context += f'\n{content}'

        task = "You are a helpful assistant and answer questions of documents. The documents are in german, so date format is usually DD.MM.YYYY."
        question = f"""Extract the effective date and the version number from the following document/text. If the effective date is not explicitly mentioned, determine the date from when the document was published.
Provide both the effective date and the version number. Sometimes the date is given as: Von/Vom Datum, Stand or as Datum des Inkrafttretens. Sometimes past versions are mentioned, so take the latest version and date.
Write the answer in the format: Publication Date: DD/MM/YYYY - Version Number: X.X.
If any of the values is not available, write the default: Publication Date: DD/MM/YYYY or Version Number: X.X.\nDocument:\n\n{context}"""
        
        answer = chat_gpt_api_call_text(task=task,question=question)
        publication_date, version_number = parse_answer(answer)
        
        # print(f'No match found for the specified pdf_name.\n\n{context}')

        # print(f'Retrieved Date: {publication_date} Version: {version_number}')

        retrieved_data = {
        "type": pdf_type,
        "file": pdf_name,
        "version": version_number,
        "date": publication_date
        }
        new_row = pd.DataFrame([retrieved_data])
        if os.path.exists(version_file):
            # If the file exists, append the new row
            df = pd.read_csv(version_file)
            df = pd.concat([df,new_row],ignore_index=True)
            df.to_csv(version_file, index=False)
        else:
            # If the file doesn't exist, create it with the new row
            df = pd.DataFrame([retrieved_data])
            df.to_csv(version_file, index=False)

    docs = []
    for n,element in enumerate(elements):
        content = element['content']
        page_element = element['page']
        type_element = element['type']
        if type_element != 'Table':
            type_element = 'Text'
            path_image = ''
        else:
            path_element = element['filePaths']
            ## Update file path and move image
            head, img_name = os.path.split(path_element)
            img_path_old = os.path.join(directory,path_element)
            img_path_new = os.path.join(file_image_storage,pdf_type,pdf_name_wo_ext,path_element)
            img_path_new = remove_part_folder_path(img_path_new,tables_folder_name)
            copy_image(img_path_old, img_path_new)
            path_image = img_path_new

        met_doc = {'doc_type':clean_text(pdf_type),
            'page': page_element,
            'type':type_element,
            'source':clean_text(pdf_name),
            'path':clean_text(path_image),
            'version':version_number,
            'valid_date':publication_date,
            'type_key':type_key,
            'file_key':file_key}
        
        doc = Document(
            page_content = content,
            metadata = met_doc
        )
        docs.append(doc)

    return docs 

def remove_part_folder_path(path,remove):

    # Split the path into directory and file name
    dir_path, filename = os.path.split(path)

    # Split the directory path into parts
    dir_parts = dir_path.split(os.sep)

    # Remove 'tables' from the path parts
    dir_parts = [part for part in dir_parts if part != remove]

    # Reconstruct the path without 'tables'
    new_dir_path = os.path.join(*dir_parts)
    new_path = os.path.join(new_dir_path, filename)

    return new_path

def copy_image(source_path, destination_path):
    """
    Copies an image from source_path to destination_path.
    
    Parameters:
    source_path (str): The path to the source image.
    destination_path (str): The path to the destination.
    """
    try:
        # Ensure the destination directory exists
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        file_name = os.path.join(destination_path,os.path.basename(source_path))
        if not os.path.exists(file_name):
        # Copy the file to the destination
            shutil.copy2(source_path, destination_path)
        # print(f"Image successfully copied from {source_path} to {destination_path}")
        
    except Exception as e:
        print(f"Error occurred while copying the image: {e}")


def create_hierarchy(elements):
    parent_stack = []
    current_parent = 0
    last_header = None
    last_table = None
    
    for element in elements:
        element_type = element['type']
        
        if element_type == 'Title':
            element['parentId'] = 0
            last_header = element['Id']
            current_parent = element['Id']
            parent_stack = [element['Id']]
            last_table = None

        elif element_type.startswith('H'):
            level = int(element_type[1:])
            while len(parent_stack) > level:
                parent_stack.pop()
            element['parentID'] = parent_stack[-1] if parent_stack else 0
            last_header = element['Id']
            current_parent = element['Id']
            parent_stack.append(element['Id'])
            last_table = None

        elif element_type == 'Table':
            element['parentID'] = last_header
            last_table = element['Id']

        elif element_type == 'Footnote':
            if last_table:
                element['parentId'] = last_table
            else:
                element['parentId'] = last_header
            last_table = None

        else:
            element['parentId'] = current_parent

    return elements

def get_content_table(elements,db_connector,pdf_file_name,pdf_file_type,db_table_name,directory):

    new_elements = []

    for i,element in enumerate(elements):
        type_element = element['type']
        if type_element == 'Table':
            table_path = element['filePaths']
            if i<len(elements)-1:
                next_type = elements[i+1]['type']
                if next_type == 'Footnote':
                    content_ft = elements[i+1]['content']
                else:
                    content_ft = ''
            else:
                content_ft = ''
            
            table_sum_dict = {'pdf_type': pdf_file_type,
                              'pdf_name': pdf_file_name,
                              'table_path': table_path}
            # print(f'table Path: {table_path}\n{table_sum_dict}')
            ids = db_connector.find_record_id(db_table_name,table_sum_dict)

            if ids == None:
                table_path_comp = os.path.join(directory,table_path)
                summary = gpt_call_table_description(table_path_comp,content_ft)
                # summary = ''
                table_sum_dict['description'] = summary
                table_sum_dict['footnote'] = content_ft
                db_connector.insert_record(db_table_name,table_sum_dict)
            
            else:
                #LOAD ID
                record = db_connector.get_record_by_id(db_table_name,ids)
                summary = record[4]

            # save_table_description()
            element['content'] = summary
            new_elements.append(element)
        elif type_element == 'Footnote':
            continue
        else:
            new_elements.append(element)
            
    return elements


def gpt_call_table_description(file_path,footnote):

    question = """Please analyze the provided image of a table and create a concise description in German that includes the following details:

	1.	Overview: Mention the primary manufacturers and the types of models listed in the table.
	2.	Model Variations: Identify any significant variations in the models, such as different series or types, and use the first letters or key differentiators to clarify distinctions.
	3.	Manufacturer Diversity: List the different manufacturers included in the table.
	4.	Value Ranges: Provide the range of key technical values (e.g., heating output, efficiency) that apply to the models listed.
	5.	Certification and Unique Attributes: Mention any certifications (e.g., BAFA) or unique attributes that apply to all or most of the models.

Ensure the description is concise, clear, and in German, reflecting the language of the table data."""

    if footnote != '':
        ft_note = f'Use the following footnote to interpret the image: {footnote}'
    else:
        ft_note = ''

    description = chat_gpt_api_call_image(img_path= file_path,question= question,footnote=ft_note)

    return description


def chat_gpt_api_call_image(img_path,question = 'Describe the following image',footnote=''):

    base64_image = encode_image(img_path)

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": question},
            {"role": "user", "content": [
                {"type": "text", "text": f"Summarize the following image.\n{footnote}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    summary = response
    summary = summary.choices[0].message.content

    return summary

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def chat_gpt_api_call_text(question,task = 'You are a helpful assistant'):

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": task},
            {"role": "user", "content": [
                {"type": "text", "text": question}
            ]}
        ],
        temperature=0.0,
    )
    summary = response
    summary = summary.choices[0].message.content

    return summary

# Function to parse the answer and extract the date and version number
def parse_answer(answer):
    date_pattern = r"Publication Date: (\d{2}/\d{2}/\d{4})"
    version_pattern = r"Version Number: (\d+\.\d+)"
    
    date_match = re.search(date_pattern, answer)
    version_match = re.search(version_pattern, answer)
    
    publication_date = date_match.group(1) if date_match else "DD/MM/YYYY"
    version_number = version_match.group(1) if version_match else "X.X"
    
    return publication_date, version_number


def chat_gpt_api_call_image_multiple(paths,instructions = 'You are a helpful assistant',question='Summarize the following image:'):

    content = []
    question_text = {'type':'text','text':question}

    dict_images = []
    for i in paths:
        encoded_img = encode_image(i)
        dic_images = {'type':'image_url','image_url':{'url': f"data:image/png;base64,{encoded_img}"}}
        dict_images.append(dic_images)

    content.append(question_text)

    if len(paths) > 0:
        content.extend(dict_images)
    

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": content}
        ],
        temperature=0.0,
    )
    summary = response
    summary = summary.choices[0].message.content

    return summary


"""

CSV PDF Processing

"""

default_conversation = [
    {"role": "system",
     "content": '''You are a text classifier. Explain why you chose that classification.'''}
]

def process_directory_csv(output_directory,db_connection,metadata_directory='data/documents/metadata',db_doc_name = table_documents_name):
    
    original_directory = output_directory.replace('output','original')
    pdf_path = original_directory+'.pdf'
    version_file = os.path.join(metadata_directory,'Files_date_version.csv')

    csv_version = pd.read_csv(version_file)

    db_con = db_connection

    name_pdf = os.path.basename(pdf_path)
    matching_row = csv_version.loc[csv_version['file'] == name_pdf]
    if not matching_row.empty:
        version_number = matching_row['version'].iloc[0]
        publication_date = matching_row['date'].iloc[0]
        type_key = int(matching_row['type_key'].iloc[0])
        file_key = int(matching_row['file_key'].iloc[0])
        
    else:
        version_number = 'N/A'
        publication_date = 'N/A'
    name = os.path.splitext(name_pdf)
    type_pdf = os.path.basename(os.path.dirname(pdf_path))
    dict = {'name':name,
            'pdf_path':pdf_path,
            'pdf_name': name_pdf,
            'type': type_pdf,
            'version' : version_number,
            'publication_date' : publication_date}

    csv_file_roots = os.path.join(output_directory,'tables')
    if os.path.exists(csv_file_roots):
        csv_clean_file = clean_csv(csv_file_roots)
        dict['csv'] = csv_clean_file

    elements = []
    elements.extend(process_pdf(dict['pdf_path']))
    elements.extend(process_csv(dict['csv']))

    pdf_file_name = dict['pdf_name']
    pdf_file_type = dict['type']

    elements = update_metadata(elements = elements,
                                pdf_type=pdf_file_type,
                                pdf_name=pdf_file_name,
                                version_number=version_number,
                                publication_date=publication_date,
                                type_key=type_key,file_key=file_key)
        
    store_documents(elements,db_con,db_doc_name,pdf_file_name,pdf_file_type)

    return elements

def update_metadata(elements,pdf_type,pdf_name,version_number,publication_date,type_key,file_key):

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

        met_doc = {'doc_type':clean_text(pdf_type),
            'page': page_element,
            'type':type_element,
            'source':clean_text(pdf_name),
            'path':"",
            'version':version_number,
            'valid_date':publication_date,
            'type_key':type_key,
            'file_key':file_key}
        
        element.metadata = met_doc

    return elements

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
        loader_csv = CSVLoader(file_path=csv_file)
        data_csv = loader_csv.load()

        return data_csv
    else:
        return []
    
def clean_csv(csv_directory,processed_csv_name = 'clean_table.csv'):

    # Process if exist
    if os.path.exists(csv_directory):

        target_location = os.path.dirname(csv_directory)

        output_name = os.path.join(target_location,processed_csv_name)

        if not os.path.exists(output_name):
            df_processed = clean_csv_file(csv_directory)
            df_processed = add_main_topic_column(df_processed)
            df_processed.to_csv(output_name, sep=',', index=False, header=True)

        return output_name
    
    else:
        return 'Not Found'

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
    # for key, value in data.items():
        # print(f"{key}:{len(value)}")
    data_frame_final = pd.DataFrame(data)

    return data_frame_final

def merge_tables(csv_list):
    for index,file in enumerate(csv_list):
        # print(file)
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
                    # print(f'{q_no} \n {text_no}')
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
        
        # print(f'DF shape {result_df.shape}')

    return result_df

def find_last_index(lst, text):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == text:
            return i
    return -1

def clean_spaces(text):
    if isinstance(text,str):
        cleaned_txt = re.sub(r'\s+',' ',text)
        cleaned_txt = cleaned_txt.strip()
        return cleaned_txt
    else:
        return text

def clean_spaces_list(list):
    return [clean_spaces(text) for text in list]

def clean_data_frame(df):

    columns = clean_spaces_list(list(df))
    # print(f'Header: {columns}')
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

# Define the cleaning function
def clean_text(text):
    # Normalize the text to combine characters like 'a' + '\u0308' into 'ä'
    normalized_text = unicodedata.normalize('NFC', text)
    return normalized_text.strip()