from utils.document_cleaning import pre_clean_text,post_clean_document
from utils.tokens_lemmatization import extract_tokens_lemm_stop,clean_extracted_values_to_tokens
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
import json
import re
from collections import defaultdict,Counter
import openai
import pandas as pd
from collections import Counter
import Levenshtein
import tqdm


API_KEY_CGPT = "sk-proj-FrNMaLZT7tgBftIDPuBOT3BlbkFJFND9eOXtUMeCyWxplFlV"
openai.api_key = API_KEY_CGPT

# Config and initialization
config = {
    'user': 'root',
    'password': 'admin123',
    'host': '127.0.0.1'
}
db_name = 'data_rag'
table_documents_name = 'table_documents'
table_retrieval_schema = {
'id': 'varchar(10) NOT NULL PRIMARY KEY',
'content': 'longtext NOT NULL',
'metadata': 'longtext NOT NULL'
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
    
    records = records[:500]
    
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
    Processes document chunks to generate embeddings and uploads them to Pinecone.

    Args:
        chunks (list): List of chunks, each containing 'id', 'content', 'metadata', and 'sparse_vector'.
        pinecone_connector (PineconeDBConnectorHybrid): Pinecone connector instance.
        embedding_handler (EmbeddingHandler): Embedding handler for generating dense embeddings.
        use_sparse (bool): Whether to include sparse vectors.
        batch_size_embedding (int): Number of chunks to process in each embedding batch.
        batch_size_upsert (int): Number of vectors to upload to Pinecone in each upsert batch.
    """
    total_chunks = len(chunks)  # Total number of chunks
    progress_bar = tqdm(total=total_chunks, desc="Processing and Uploading Chunks")

    for i in range(0, total_chunks, batch_size_embedding):
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

        # Update the progress bar
        progress_bar.update(len(batch))

    progress_bar.close()
    print("All batches uploaded successfully.")