import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import os
import base64
import requests
from dotenv import load_dotenv
import openai
from IPython.display import Image, display, Audio, Markdown
from langchain_core.documents import Document
import pandas as pd
import shutil


load_dotenv()
API_KEY_CGPT = os.getenv('API_KEY_CGPT')
print(f'Loaded API KEY ChatGPT: {API_KEY_CGPT}')
openai.api_key = API_KEY_CGPT
MODEL="gpt-4o"


def copy_image(source_path, destination_path):
    """
    Copies an image from source_path to destination_path.
    
    Parameters:
    source_path (str): The path to the source image.
    destination_path (str): The path to the destination.
    """
    try:
        shutil.copy2(source_path, destination_path)
        # print(f"Image successfully copied from {source_path} to {destination_path}")
    except Exception as e:
        print(f"Error occurred while copying the image: {e}")

# Function to display image from file path
def display_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()


# Load the JSON file
def read_json(file_read):
    with open(file_read, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to extract and categorize elements from the JSON structure
def extract_and_categorize_elements(json_data):
    extracted_items = []
    current_table = None
    pattern = r'^\d+ \w+( \w+)*'
    pattern_sub = r'^\d+\.\d+ .+'

    if 'elements' in json_data:
        for element in json_data['elements']:
            path = element.get('Path', '')
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
                            'type': 'table',
                            'filePaths': element['filePaths'],
                            'page':page
                        })
            else:
                # If it's a non-table element, reset current_table
                current_table = None

                # # Categorize text elements
                # if 'Title' in path:
                #         item_type = 'title'
                # elif bool(re.match(pattern=pattern,string=text)) and 'Bd_0' in font_name:
                #         item_type = 'chapter'
                # elif bool(re.match(pattern=pattern_sub,string=text)):
                #         item_type = 'subchapter'
                # elif 'List' in path or 'Bullet' in path or 'Item' in path:
                #     item_type = 'list_item'
                # else:
                item_type = 'text'

                # Store the categorized element
                extracted_items.append({
                    'type': item_type,
                    'content': text.strip(),
                    'textSize': text_size,
                    'font': font_name,
                    'page':page
                })

    return extracted_items

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

# Function to extract and categorize elements from the JSON structure
def extract_and_categorize_elements2(json_data):
    extracted_items = []
    current_table = None
    pattern = r'^\d+ \w+( \w+)*'
    pattern_sub = r'^\d+\.\d+ .+'

    if 'elements' in json_data:
        for element in json_data['elements']:
            path = element.get('Path', '')
            text = element.get('Text', '')
            text_size = element.get('TextSize', 0)
            font_name = element.get('Font', {}).get('name', '')
            page = element.get('Page', '')
            item_type = get_last_element(path)


            # Check if the current element is part of a table
            if 'Table' in path:
                # If it's a new table, record it
                if current_table != path:
                    current_table = path
                    if 'filePaths' in element:
                        extracted_items.append({
                            'type': item_type,
                            'filePaths': element['filePaths'],
                            'page':page
                        })
            else:
                # If it's a non-table element, reset current_table
                current_table = None

                extracted_items.append({
                    'type': item_type,
                    'content': text.strip(),
                    'textSize': text_size,
                    'font': font_name,
                    'page':page
                })

    return extracted_items

# Function to get the category of a type
def get_category(types,type_name):
    for category, type_list in types.items():
        if type_name in type_list:
            return category
    return 'other'

def merge_correct_elements(elements):

    list_types = ['Lbl','LBody','L','Li']
    paragraph_types = ['P','ParagraphSpan']
    tables_parts = ['Table','Footnote']
    titles_headings = ['Title','H1','H2','H3','H4','H5','H6']
    types = {'list':list_types,'paragraph':paragraph_types,
             'table':tables_parts,'titles':titles_headings}
    previous_type = None

    for i,element in enumerate(elements):
        type = element['type']
        cat = get_category(types,type)

        if type != 'Table':
            content = element['content']
            page = element['page']
            textSize = element['textSize']
            font = element['font']
            print(f'Element: {type} - Category: {cat} - page: {page} content: {content[:10]}\n')
        
        else:
            file_paths = element['filePaths']
            page = element['page']

            print(f'Element: {type} - page: {page} content: {file_paths}\n')

    return elements

def chunk_document(elements):
    chunks = []
    current_chunk = None

    for element in elements:
        if element['type'] == 'Title':
            if current_chunk:
                chunks.append(current_chunk)
            else:
                current_chunk = {'level': 'top', 'type': 'Title', 'content': element['content'], 'elements': [element]}
        elif element['type'] in ['H1', 'H2', 'H3']:
            if current_chunk:
                chunks.append(current_chunk)
            else:
                current_chunk = {'level': 'middle', 'type': element['type'], 'content': element['content'], 'elements': [element]}
        elif element['type'] == 'P':
            if current_chunk and current_chunk['level'] == 'middle':
                current_chunk['elements'].append(element)
            else:
                chunks.append({'level': 'bottom', 'type': 'P', 'content': element['content'], 'elements': [element]})
        # Handle other types similarly...

    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def process_json(directory):
    files = os.listdir(directory)
    files = [i for i in files if i.endswith('json')]
    print(files)
    for i_file in files:
        file = os.path.join(directory,i_file)
        data = read_json(file)
        elements = extract_and_categorize_elements2(data)
        elements = merge_correct_elements(elements)
        # chunks = chunk_document(elements)

        # elements2 = merge_elements(elements)
        # print_extracted_elements(elements2,directory)
        return elements


# Function to merge elements based on patterns and matching font/size
def merge_elements(elements):
    pattern = re.compile(r'^[a-zA-Z]\)|^-$|^–$')
    merged_elements = []
    i = 0
    while i < len(elements):
        element = elements[i]
        if element['type'] == 'text' and pattern.match(element['content']):
            if i + 1 < len(elements):
                next_element = elements[i + 1]
                if (element['textSize'] == next_element['textSize'] and
                        element['font'] == next_element['font'] and
                        element['page'] == next_element['page']):
                    # Merge current element with next element
                    merged_content = element['content'] + " " + next_element['content']
                    merged_elements.append({
                        "type": "text",
                        "content": merged_content,
                        "textSize": element['textSize'],
                        "font": element['font'],
                        "page": element['page']
                    })
                    i += 2  # Skip the next element
                    continue
        elif element['type'] == 'table':
            tables = [i for i in element['filePaths'] if i.endswith('.png')]
            print(f'Tabs: {tables}')
            for j,table in enumerate(tables):
                element_copy = element.copy()
                element_copy['tablePath'] = table
                element_copy['page'] = int(element_copy['page'])+j
                if j+1 == len(tables):
                    element = element_copy.copy()
                    print(f'original: {element}')
                else:
                    merged_elements.append(element_copy)
                    print(f'copia: {element_copy}')
        merged_elements.append(element)
        i += 1
    return merged_elements

def print_extracted_elements(extracted_items,base_path_img):
    # Print the extracted items with their categories
    for item in extracted_items:
        if item['type'] != 'table':
            print(f"{item['type']}\nFont:{item['font']} - Size:{item['textSize']}\nContent: {item['content']}\nPage: {item['page']}\n")
        else:
            print(f"Table Path: {item['tablePath']}")
            file_path = item['tablePath']
            if file_path.endswith('png'):
                path_img = os.path.join(base_path_img,file_path)
                page_item = item['page']
                print(f'Path: {path_img} - Page start: {page_item}\n')
                display_image(path_img)




def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def summarise_tables_cgpt(extracted_items,base_path_img):
    # Print the extracted items with their categories
    new_extracted_items = []
    for item in extracted_items:
        if item['type'] == 'table':
            # print(f"Table Path: {item['tablePath']}")
            file_path = item['tablePath']
            if file_path.endswith('png'):
                path_img = os.path.join(base_path_img,file_path)
                # print(f'Path: {path_img}')
                # display(Image(path_img))
                base64_image = encode_image(path_img)
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": """Please provide a concise summary of the following table in German. Highlight the key categories and their associated values, focusing on any significant numbers or ranges. Ensure the summary is clear, concise, and suitable for embedding in a retrieval system."""},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Summarize the following image:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]}
                    ],
                    temperature=0.0,
                )
                summary = response
                summary = summary.choices[0].message.content
            item['summary'] = summary
        new_extracted_items.append(item)
    return new_extracted_items

def join_pages_create_document(elements,file_name,file_type,version,valid_date,act_img_directory,storage_img_dir):
    documents = []
    act_page = None
    create_doc = False
    for i,element in enumerate(elements):
        page_element = element['page']
        type_element = element['type']
        if type_element == 'text':
            if page_element != act_page:
                if i > 0 and elements[i-1]['type']=='text':
                    doc = Document(
                        page_content = content,
                        metadata = met_doc
                    )
                    documents.append(doc)
                content = element['content']
                met_doc = {'doc_type':file_type,
                           'page': page_element,
                           'type':type_element,
                           'source':file_name,
                           'path':'',
                           'version':version,
                           'valid_date':valid_date}
                act_page = page_element
            else:
                content = content + '\n' + element['content']
                if i == len(elements)-1:
                    create_doc = True
        elif type_element == 'table':
            if i > 0 and elements[i-1]['type']=='text':
                doc = Document(
                page_content = content,
                metadata = met_doc
                )
                documents.append(doc)
            content = element['summary']
            # Copy img to new location
            head, img_name = os.path.split(element['tablePath'])
            img_path_old = os.path.join(act_img_directory,element['tablePath'])
            img_name_new = file_name+'_'+img_name
            img_path_new = os.path.join(storage_img_dir,img_name_new)
            copy_image(img_path_old, img_path_new)
            met_doc = {'doc_type':file_type,
                        'page': page_element,
                        'type':type_element,
                        'source':file_name,
                        'path':img_path_new,
                        'version':version,
                        'valid_date':valid_date}
            create_doc = True
            act_page = None
        if create_doc:
            doc = Document(
                page_content = content,
                metadata = met_doc
            )
            documents.append(doc)
            create_doc = False

    return documents


def get_pdf_json(docs_directory,target_file,storage_directory):

    original_directory = os.path.join(docs_directory,'original')
    metadata_directory = os.path.join(docs_directory,'metadata')
    output_directory = os.path.join(docs_directory,'output')
    storage_img_directory = os.path.join(storage_directory,'images')

    version_file = os.path.join(metadata_directory,'files_date_version.csv')
    csv_version = pd.read_csv(version_file)

    files = os.listdir(os.path.join(original_directory,target_file))
    pdf_files = [i for i in files if i.endswith('pdf')]
    print(pdf_files)
    datas = []
    json_file_name = 'structuredData.json'
    for i,file in enumerate(pdf_files):
        file_name, file_extension = os.path.splitext(file)
        pdf_file = os.path.join(original_directory,target_file,file)
        json_file_directory = os.path.join(output_directory,target_file,file_name)
        json_file = os.path.join(json_file_directory,json_file_name)

        # print(f'File: {file} - pdf: {pdf_file} json: {json_file}')
        if os.path.exists(pdf_file):
            # print('PDF exists\n')
            if os.path.exists(json_file):
                # print('JSON exists\n')
                version_i = csv_version.loc[csv_version['file'] == file, 'version'].iloc[0]
                date_i = csv_version.loc[csv_version['file'] == file, 'date'].iloc[0]
                print(f'Version: {version_i} - date: {date_i}\n')

                data = read_json(json_file)
                elements = extract_and_categorize_elements(data)
                elements2 = merge_elements(elements)
                elements3 = summarise_tables_cgpt(elements2,json_file_directory)
                documents = join_pages_create_document(elements3,file_name,target_file,version_i,date_i,json_file_directory,storage_img_directory)
                datas.append(documents)
    
    return datas

def process_json_images(directory):
    # Split the path into head (directory) and tail (file name)
    head, tail = os.path.split(directory)
    # Extract the directory name
    file_type = os.path.basename(head)
    # Extract the file name
    file_name = tail
    files = os.listdir(directory)
    files = [i for i in files if i.endswith('json')]
    print(files)
    for i_file in files:
        file = os.path.join(directory,i_file)
        data = read_json(file)
        elements = extract_and_categorize_elements(data)
        elements2 = merge_elements(elements)
        elements3 = summarise_tables_cgpt(elements2,directory)
        documents = join_pages_create_document(elements3,file_name,file_type,version,valid_date)
        return documents

def get_json_elements(directory):
    files = os.listdir(directory)
    files = [i for i in files if i.endswith('json')]
    print(files)
    for i_file in files:
        file = os.path.join(directory,i_file)
        data = read_json(file)
        elements = extract_and_categorize_elements(data)
        elements2 = merge_elements(elements)
        return elements2

""" 

# Function to encode the image


## Set the API key and model name


openai.api_key = API_KEY_CGPT


# Path to your image
image_path = "output/Richtlinie BEG EM/Richtlinie_BEG_EM_(2021-05-20)/tables/fileoutpart4.png"
display(Image(image_path))

# Getting the base64 string
base64_image = encode_image(image_path)


response = openai.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": Please provide a concise summary of the following table in German. Highlight the key categories and their associated values, focusing on any significant numbers or ranges. Ensure the summary is clear, concise, and suitable for embedding in a retrieval system.},
        {"role": "user", "content": [
            {"type": "text", "text": "Summarize the following image:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content) """