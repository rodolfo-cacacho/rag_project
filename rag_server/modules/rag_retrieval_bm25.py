import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
from collections import Counter,defaultdict
import os
import re
import spacy
import nltk
from nltk.corpus import stopwords
from utils.document_cleaning import post_clean_document
from utils.tokens_lemmatization import extract_tokens_lemm_stop,clean_extracted_values_to_tokens
import Levenshtein


# Load German stopwords and the German spaCy model
nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))
nlp = spacy.load("de_core_news_sm")


def get_dates_documents(input_date, date_database):
    # print(f'The input date is the following: {input_date}')
    # Convert the date column to datetime format for proper filtering
    date_database['date'] = pd.to_datetime(date_database['date'], format="%d/%m/%Y", errors='coerce', dayfirst=True)
    # Filter out rows where the date is NaT (i.e., invalid dates)
    date_database = date_database.dropna(subset=['date'])

    # Parse the input date
    input_date = pd.to_datetime(input_date, dayfirst=False)
    doc_types = date_database[['type', 'type_key']].drop_duplicates()
    doc_types = doc_types.to_dict(orient='records')


    for i in doc_types:
        i['logic'] = get_logic_date(i['type'])

    cont_None = 0
    # Iterate over each document type
    for i, doc_type in enumerate(doc_types):
        # Filter the dataframe based on the document type
        filtered_df = date_database[date_database['type'] == doc_type['type']]
        # Further filter to include only documents before or on the input date
        filtered_df = filtered_df[filtered_df['date'] <= input_date].sort_values(by='date', ascending=False)

        search_type = doc_type['logic']

        if search_type == 'latest':
            # Check if the most recent document is an "Änderung"
            file_text = filtered_df.iloc[0]['file'].lower() if not filtered_df.empty else 'empty'
            text_changes = "änderung"
            if file_text != 'empty' and text_changes in file_text:
                # Collect all "Änderung" documents
                # print(f"Änderung {filtered_df.iloc[0]['file']}")
                all_docs = []
                for index, row in filtered_df.iterrows():
                    all_docs.append(row['date'].strftime('%d/%m/%Y'))
                    if text_changes not in row['file'].lower():
                        break
                if all_docs == []:
                    cont_None+=1
                doc_type['dates'] = list(reversed(all_docs))  # Reverse to have them in chronological order
            else:
                # Just return the latest document
                if not filtered_df.empty:
                    recent_doc = filtered_df.head(1)
                    formatted_dates = recent_doc['date'].dt.strftime('%d/%m/%Y').tolist()
                    doc_type['dates'] = formatted_dates
                else:
                    cont_None+=1
                    doc_type['dates'] = []
        
        elif search_type == 'all':
            # Find all documents on or before the input date
            all_docs = filtered_df.sort_values(by='date', ascending=True)
            formatted_dates = all_docs['date'].dt.strftime('%d/%m/%Y').tolist()
            if formatted_dates == []:
                cont_None +=1
            doc_type['dates'] = formatted_dates
        
        else:
            cont_None +=1
            doc_type['dates'] = []

    if cont_None == len(doc_types):
        return None
    else:
        return doc_types

def get_logic_date(document_class):
    
    class_new = ['Technische FAQ BEG','BMWK - FAQ BEG','Richtlinie BEG EM','FAQ BEG (EM und EH-EG)',
                 'Förderübersicht BEG EM','Allgemeines Merkblatt zur Antragstellung','Infoblätter förderfähigen Kosten',
                 'Liste förderfähigen Anlagen - Biomasse','Liste förderfähigen Anlagen - Wärmepumpen']

    if document_class in class_new:
        return 'latest'
    else:
        return 'all'


# Re-ranking function with a parameter to limit results after re-ranking
def rerank_documents_by_distance_and_model(documents, query, max_results):
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-reranker-base",trust_remote_code = True)

    # Sort documents by distance first (but don't filter based on max_results yet)
    documents_sorted_by_distance = sorted(documents, key=lambda x: x['distance'])

    # Assign original ranking based on distance
    for rank, doc in enumerate(documents_sorted_by_distance, start=1):
        doc['org_ranking'] = rank

    # Extract content for re-ranking
    passages = [doc['content'] for doc in documents_sorted_by_distance]

    # Encode the query and the passages
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)

    # Compute cosine similarity between the query and each passage
    cosine_scores = util.cos_sim(query_embedding, passage_embeddings)

    # Re-rank based on similarity scores
    ranked_documents = sorted(zip(documents_sorted_by_distance, cosine_scores[0].tolist()), key=lambda x: x[1], reverse=True)

    # Assign new ranking based on the re-ranker
    for new_rank, (doc, score) in enumerate(ranked_documents, start=1):
        doc['new_ranking'] = new_rank

    # Limit the number of documents after re-ranking
    limited_ranked_documents = ranked_documents[:max_results]

    # Track how many documents are filtered out
    filtered_out_count = len(ranked_documents) - len(limited_ranked_documents)

    return [doc for doc, _ in limited_ranked_documents], filtered_out_count


def preprocess_vocabulary(vocabulary_list):
    """
    Converts a list of vocabulary dictionaries into a dictionary for efficient lookup,
    including synonyms for each word.

    Parameters:
    - vocabulary_list: list of dictionaries, each containing 'id', 'word', 'idf', and 'synonyms'

    Returns:
    - dict: dictionary where keys are words and their synonyms (lowercased),
            and values are dictionaries with 'id', 'idf', and original word.
    """
    vocab_dict = {}
    for entry in vocabulary_list:
        word = entry['word'].lower()
        idf = entry['idf']
        word_id = entry['id']
        synonyms = json.loads(entry['synonyms']) # Convert string representation of the list into an actual list
        
        # Add the original word
        vocab_dict[word] = {'id': word_id, 'idf': idf, 'original_word': word}
        
        # Add synonyms with the same IDF and ID
        for synonym in synonyms:
            synonym_lower = synonym.lower()
            if synonym_lower not in vocab_dict:  # Avoid overwriting existing entries
                vocab_dict[synonym_lower] = {'id': word_id, 'idf': idf, 'original_word': word}
    
    return vocab_dict


def bm25_query_vectors(queries, vocab_dict, k1=1.5, similarity_threshold=0.90,n_grams = [2,3]):
    """
    Generates BM25 sparse vectors for multiple queries with fuzzy matching.

    Parameters:
    - queries: list of str, the input queries.
    - vocab_dict: dict, preprocessed vocabulary dictionary with words as keys and {'id': id, 'idf': idf} as values.
    - k1: float, BM25 term frequency saturation parameter (default=1.5).
    - similarity_threshold: int, minimum similarity score (0-100) for fuzzy matching (default=80).

    Returns:
    - sparse_vectors: list of dict, each sparse BM25 vector in the format {'indices': [...], 'values': [...]}.
    """

    def create_query_ngrams(tokens, n_values):
        """
        Generates n-grams of specified sizes from a list of tokens.

        Parameters:
        - tokens: list of str, preprocessed tokens from the query.
        - n_values: list of int, the sizes of n-grams to generate (e.g., [2, 3] for 2-grams and 3-grams).

        Returns:
        - list of str, the combined list of n-grams for all specified sizes.
        """
        ngrams = []
        for n in n_values:
            ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        # print(ngrams)
        return ngrams

    def fuzzy_match(term, vocab_dict, threshold=0.50):
        """
        Performs fuzzy matching for a term in the vocabulary dictionary using normalized words.

        Parameters:
        - term: str, the query term to match.
        - vocab_dict: dict, the vocabulary dictionary with terms as keys.
        - threshold: float, minimum similarity score (0-1) for a match.

        Returns:
        - match: str or None, the best matching term from the vocabulary or None if no match meets the threshold.
        """
        def normalize_word(word):
            """Normalizes umlauts and removes hyphens."""
            return word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss").replace("-", "")

        # Normalize the input term
        normalized_term = normalize_word(term)

        best_match = None
        highest_score = 0

        for vocab_term in vocab_dict.keys():
            # Normalize the vocabulary term
            normalized_vocab_term = normalize_word(vocab_term)

            # Calculate Levenshtein similarity
            similarity = Levenshtein.ratio(normalized_term, normalized_vocab_term)

            # Update the best match if the similarity is higher than the current threshold and highest score
            if similarity >= threshold and similarity > highest_score:
                best_match = vocab_term
                highest_score = similarity

        return best_match

    def process_query(query):
        """Generates a BM25 sparse vector for a single query."""
        # Step 1: Preprocess and tokenize the query
        clean_query, query_terms = post_clean_document(query)  # Tokenize and preprocess the query
        tokens_query = extract_tokens_lemm_stop(clean_query)
        ngrams = create_query_ngrams(tokens_query,n_grams)
        query_terms = clean_extracted_values_to_tokens(query_terms)
        query_tokens = tokens_query + query_terms + ngrams

        # Step 2: Count the frequency of each query token
        term_frequencies = Counter(query_tokens)

        # Step 3: Prepare indices and values for the sparse vector
        indices = []
        values = []
        processed_ids = set()  # To avoid duplicates for terms/synonyms pointing to the same ID

        for term, tf in term_frequencies.items():
            # Perform exact or fuzzy matching
            match = term.lower() if term.lower() in vocab_dict else fuzzy_match(term.lower(), vocab_dict, similarity_threshold)
            if match:
                vocab_entry = vocab_dict[match]
                term_id = vocab_entry['id']  # Get the term ID
                idf = vocab_entry['idf']  # Get the term's IDF value

                # Calculate the BM25 score for this term
                bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1))

                # Add the term ID and its score if not already processed
                if term_id not in processed_ids:
                    indices.append(term_id)
                    values.append(bm25_score)
                    processed_ids.add(term_id)

        # Step 4: Return the sparse vector
        return {'indices': indices, 'values': values}

    # Process all queries and generate sparse vectors
    sparse_vectors = [process_query(query) for query in queries]
    return sparse_vectors

def bm25_query_vectors_original(queries, vocab_dict, k1=1.5):
    """
    Generates BM25 sparse vectors for multiple queries.

    Parameters:
    - queries: list of str, the input queries.
    - vocab_dict: dict, preprocessed vocabulary dictionary with words as keys and {'id': id, 'idf': idf} as values.
    - k1: float, BM25 term frequency saturation parameter (default=1.5).

    Returns:
    - sparse_vectors: list of dict, each sparse BM25 vector in the format {'indices': [...], 'values': [...]}.
    """

    def process_query(query):
        """Generates a BM25 sparse vector for a single query."""
        # Step 1: Preprocess and tokenize the query
        clean_query, query_terms = post_clean_document(query)  # Tokenize and preprocess the query
        tokens_query = extract_tokens_lemm_stop(clean_query)
        query_terms = clean_extracted_values_to_tokens(query_terms)
        query_tokens = tokens_query + query_terms

        # Step 1: Count the frequency of each query token
        term_frequencies = Counter(query_tokens)

        # Step 2: Prepare indices and values for the sparse vector
        indices = []
        values = []
        processed_ids = set()  # To avoid duplicates for terms/synonyms pointing to the same ID

        for term, tf in term_frequencies.items():
            if term in vocab_dict:  # Check if the term or its synonym exists in the vocabulary
                vocab_entry = vocab_dict[term]
                term_id = vocab_entry['id']  # Get the term ID
                idf = vocab_entry['idf']  # Get the term's IDF value

                # Calculate the BM25 score for this term
                bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1))

                # Add the term ID and its score if not already processed
                if term_id not in processed_ids:
                    indices.append(term_id)
                    values.append(bm25_score)
                    processed_ids.add(term_id)

        # Step 3: Return the sparse vector
        return {'indices': indices, 'values': values}

    # Process all queries and generate sparse vectors
    sparse_vectors = [process_query(query) for query in queries]
    return sparse_vectors

def hybrid_scale(dense, sparse, alpha: float = 1.0):
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse["indices"],
        'values':  [val * (1 - alpha) for val in sparse["values"]]
    }
    hdense = [val * alpha for val in dense]
    return hdense, hsparse

def retrieve_context(vector_db_connector,sql_connector,embed_handler,
                     embed_task,sql_table,alpha_value,
                     vocab_table,query,gen_prompts,
                     keyterms,max_results,retrieve_date_documents,
                     max_results_query,distance_threshold = 0.5):
    
    distance_threshold = 0

    or_condition = {"$or": []}
    for i in retrieve_date_documents:
        doc = i['type']
        dates = i['dates']
        if len(dates)>0:
            type_key = i['type_key']
            temp_and = {"$and": [{"valid_date":{'$in':dates}},{"type_key":{'$eq':type_key}}]}
            if len(retrieve_date_documents) > 1:
                or_condition["$or"].append(temp_and)
            else:
                or_condition = temp_and

    # print("Reading Vocabulary")

    vocabulary_bm25 = sql_connector.get_all_records_as_dict(vocab_table)
    vocabulary_bm25 = preprocess_vocabulary(vocabulary_bm25)

    # print("Done reading vocabulary")

    # print(f"{gen_prompts} {len(gen_prompts)}")
    # print(query)

    queries = [query] + gen_prompts
    # print(f"{queries} {len(queries)}")

    # print("Sparse vectors")
    sparse_vectors = bm25_query_vectors(queries,vocabulary_bm25)
    # print("Sparse vectors done")

    dense_vectors = embed_handler.embed_texts(texts = queries,
                                              task = embed_task)
    # print("Dense vectors done")

    results_dict = {}
    id_count = defaultdict(int)

    # Step 1: Collect scores and counts during query processing
    for i, query in enumerate(queries):
        dense_vector, sparse_vector = hybrid_scale(
            dense=dense_vectors[i],
            sparse=sparse_vectors[i],
            alpha=alpha_value
        )
        results = vector_db_connector.query_collection(
            query_embedding_dense=dense_vector,
            query_embedding_sparse=sparse_vector,
            n_results=max_results_query,
            where=or_condition
        )

        for match in results['matches']:
            match_id = match['id']
            match_score = match['score']
            id_count[match_id] += 1

            # Store scores for each match_id across queries
            if match_id not in results_dict:
                results_dict[match_id] = []
            results_dict[match_id].append(match_score)

    # Step 2: Normalize scores
    # Flatten all scores to find the global min and max
    all_scores = [score for scores in results_dict.values() for score in scores]
    global_min_score = min(all_scores)
    global_max_score = max(all_scores)

    # Normalize each score for each chunk
    for match_id, scores in results_dict.items():
        results_dict[match_id] = [
            (score - global_min_score) / (global_max_score - global_min_score) for score in scores
        ]

    # Step 3: Calculate final scores (Hybrid: Max + Average)
    w1, w2 = 0.7, 0.3  # Weights for max and average scores
    final_scores = {}
    for match_id, scores in results_dict.items():
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        final_scores[match_id] = w1 * max_score + w2 * avg_score

    # Step 4: Apply frequency boost (optional)
    beta = 0.1
    for match_id in final_scores.keys():
        if id_count[match_id] > 1:
            final_scores[match_id] += beta * (id_count[match_id] - 1)

    # Step 5: Sort results by final adjusted score
    ranked_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # Display results
    # for match_id, score in ranked_results:
    #     print(f"Chunk ID: {match_id}, Final Score: {score}")

    # Step 6: Convert results_dict to a DataFrame for easier manipulation
    df_results = pd.DataFrame(ranked_results, columns=['id', 'score'])

    # Calculate top 5 highest scores
    top_5 = df_results.nlargest(5, 'score')

    # Calculate the lowest score
    lowest = df_results['score'].min()

    # Calculate the average score
    average = df_results['score'].mean()

    # Display the results
    # print("Top 5 highest scores:")
    # print(top_5)
    # print("\nLowest score:", lowest)
    # print("Average score:", average)

    # Step 7: Find and display IDs with multiple results
    multiple_results_ids = {id: count for id, count in id_count.items() if count > 1}
    # print(f"IDs with multiple results (and their counts): {multiple_results_ids}")

    # Step 8: Prepare final ID list
    id_list = list(results_dict.keys())
    
    if len(results_dict) == 0:

        return None,None,None

    else:

        records = sql_connector.get_records_by_ids(sql_table,id_list)

        # Step 1: Convert the records list into a DataFrame
        df = pd.DataFrame(records)
        df.drop(columns=['content'], inplace=True)

        # print(f'DF: {df}')
        # Step 2: Unpack the 'metadata' column, which is a JSON string, into separate columns
        # df['metadata'] = df['metadata'].str.replace("'", '"')
        df_metadata = df['metadata'].apply(json.loads).apply(pd.Series)
        df.drop(columns=['metadata'], inplace=True)

        # Step 3: Concatenate the original DataFrame with the unpacked metadata columns
        df_combined = pd.merge(df, df_metadata, how='left', on='id')
        df_distances = pd.DataFrame(ranked_results, columns=['id', 'distance'])

        df_merged = pd.merge(df_combined, df_distances, on='id', how='left')
        df_merged['distance'] = df_merged['distance'].fillna(0)


        # Filter the DataFrame
        # Filter based on the distance threshold
        df_filtered = df_merged[df_merged['distance'] >= distance_threshold]

        # Ensure that df_filtered has at most max_results items, sorted by smallest distance
        df_filtered = df_filtered.sort_values(by='distance',ascending=False).head(max_results)

        filtered_ids = df_filtered['id'].tolist()
        # Convert the filtered DataFrame to a list of dictionaries
        filtered_data = df_filtered.to_dict(orient='records')

        filtered_data = get_ordered_records(filtered_data)
        # print(f'filtered_ids {filtered_ids}')

        if len(filtered_data)>0:
            # print(f'Filtered data: {filtered_data}')
            return filtered_data,id_list,filtered_ids
        else:
            # print('No data founded')
            return None,id_list,None

def get_ordered_records(records):
    # Sort the records based on the id
    return sorted(records, key=lambda x: tuple(map(int, x['id'].split('.'))))

def ids_to_retrieve(records,dif=1,surrounding_chunks = 1):
    context_ranges = []

    # Iterate through the list and process each entry
    for index, entry in enumerate(records):
        entry_doc, entry_chunk = entry['id'].split('.')
        pre_chunk = max(int(entry_chunk)-surrounding_chunks,0)
        _,entry_chunk_max = entry['last_id'].split('.')
        post_chunk = min(int(entry_chunk)+surrounding_chunks,int(entry_chunk_max))

        entry_range = {
            'doc': entry_doc,
            'start': int(pre_chunk),
            'end': int(post_chunk),
            'doc_original':[entry['id']]
        }
        # print(f'entry {entry_range} {context_ranges}')
        if index == 0:
            context_ranges.append(entry_range)
        
        else:
            # logica de revisar
            for n,j in enumerate(context_ranges):
                if j['doc'] == entry_range['doc']:
                    if j['start'] < entry_range['start']:
                        if j['end']+dif >= entry_range['start']:
                            # print(f'merge {j} {entry_range}')
                            j['end'] = entry_range['end']
                            j['doc_original'].append(entry_range['doc_original'][0])
                            break
                    else:
                        if entry_range['end']+dif >= j['start']:
                            # print(f'merge {j} {entry_range}')
                            j['start'] = entry_range['start']
                            j['doc_original'].append(entry_range['doc_original'][0])
                            break
                
                if n == len(context_ranges)-1:
                    context_ranges.append(entry_range)
                    break

    # Generate the list of IDs for each range
    orig_ids = []
    ranges = []
    for entry_range in context_ranges:
        doc = entry_range['doc']
        start = entry_range['start']
        end = entry_range['end']

        ids = [f"{doc}.{i}" for i in range(start, end + 1)]
        orig_ids.append(entry_range['doc_original'])
        ranges.append(ids)

    return ranges,orig_ids


def build_context(context,sql_connector,table_name):

    if context == None:
        return context

    else:
        # print(f'pre ids {context}')
        ids,original = ids_to_retrieve(context)
        # print(f'building context: {ids}')

        records = []
        for i in ids:
            # print(f'i: {i}')
            results = sql_connector.get_records_by_ids(table_name, i)
            records.append(results)

    context = []
    table_n = 1

    for in_r,record in enumerate(records):
        tab_nums = []
        context_i = {'id':'',
                'source':'',
                'pages':[],
                'text':'',
                'paths':[]}
        act_page = None
        for index,sub_rec in enumerate(record):
            metadata = json.loads(sub_rec['metadata'].replace("'",'"'))
            context_i['text'] += f"{sub_rec['content']}\n"
            context_i['pages'].append(metadata['page'])
            if index == 0:
                context_i['id'] = original[in_r]
                context_i['source'] = metadata['doc_type']
            for i_tab,table in enumerate(metadata['path']):
                if table not in context_i['paths']:
                    context_i['paths'].append(table)
                    tab_nums.append(table_n)
                    table_n+=1

        non_numeric_texts = []
        flattened_pages = []
        for entry in context_i['pages']:
            # Use regex to find numeric parts
            numbers = re.findall(r'\d+\.\d+|\d+', entry)
            
            # If any non-numeric text is present, capture it and exit the loop
            if not numbers or any(not part.isdigit() for part in entry.replace(",", "").split()):
                non_numeric_texts.append(entry)
                continue
            
            # Otherwise, extend the list with numeric parts as integers
            flattened_pages.extend(int(num) for num in numbers)

        # Check if there were any non-numeric entries
        if non_numeric_texts:
            # Join all non-numeric entries as a single string
            page_range = ', '.join(non_numeric_texts)
        else:
            # Calculate min and max if all entries were numeric
            min_page = min(flattened_pages)
            max_page = max(flattened_pages)
            page_range = f"{min_page}-{max_page}"

        context_i['pages'] = page_range
        text_tables = ''
        if len(context_i['paths'])>0:
            for n,i in enumerate(context_i['paths']):
                text_tables += f"Table {tab_nums[n]} - {context_i['source']}/{os.path.basename(i)}\n\n"
            context_i['text'] += text_tables
        
        context.append(context_i)

    return context
    
def get_data_ids(sql_connector, sql_table, ids):
    # Step 1: Fetch records from the database
    records = sql_connector.get_records_by_ids(sql_table, ids)
    
    # Step 2: Convert the records list into a DataFrame
    df = pd.DataFrame(records)
    
    # Step 3: Unpack the 'metadata' column, which is a JSON string, into separate columns
    df['metadata'] = df['metadata'].str.replace("'", '"')  # Ensure JSON is correctly formatted
    df_metadata = df['metadata'].apply(json.loads).apply(pd.Series)  # Unpack JSON into columns
    df_metadata.drop(columns=['id'], inplace=True)  # Drop the 'id' column if not needed
    
    # Step 4: Concatenate the original DataFrame with the unpacked metadata columns
    df_combined = pd.concat([df.drop(columns=['metadata']), df_metadata], axis=1)

    # Step 5: Convert the combined DataFrame to a list of dictionaries
    data = df_combined.to_dict(orient='records')
    
    # Optionally reorder the records (if required by your application)
    data = get_ordered_records(data)
    
    return data
    



