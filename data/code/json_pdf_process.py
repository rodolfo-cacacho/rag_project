import os
import json
import re
from langchain_core.documents import Document
import base64
from dotenv import load_dotenv
import openai
import pandas as pd
import shutil
from database_manager import MySQLDB

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
    db.create_table(table_name_documents, schema_documents)

    return db

def process_directory(directory,file_image_storage, max_pages=50,default_json_file = 'structuredData.json',default_tables_folder = 'tables',metadata_directory = 'data/documents/metadata',db_table_name = table_description_name,db_doc_name = table_documents_name):
    # Identify part folders and sort them
    
    pdf_file_name = os.path.basename(directory)+'.pdf'
    pdf_file_type = os.path.basename(os.path.dirname(directory))
    print(f'pdf file: {pdf_file_name} - pdf type: {pdf_file_type}')

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

    db_con = init_db_connection()

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
        print(f'reading file: {json_file_path}')
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
                        'metadata':json.dumps(i.metadata)}
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
        print(f'Version: {version_number} - Date: {publication_date}\n')
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

        print(f'Retrieved Date: {publication_date} Version: {version_number}')

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

        met_doc = {'doc_type':pdf_type,
            'page': page_element,
            'type':type_element,
            'source':pdf_name,
            'path':path_image,
            'version':version_number,
            'valid_date':publication_date}
        
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
            print(f'table Path: {table_path}\n{table_sum_dict}')
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

