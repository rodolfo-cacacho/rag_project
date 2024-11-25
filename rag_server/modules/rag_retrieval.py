import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util


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

def retrieve_context(vector_db_connector,sql_connector,sql_table,query,gen_prompts,keyterms,max_results,retrieve_date_documents,max_results_query,distance_threshold_query,distance_threshold = 0.5):

    # print('Generated prompts')
    # for i in gen_prompts:
    #     print(f'{i}')
    # print('Extracted Keyterms')
    # for i in keyterms:
    #     print(f'{i}')

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

    results = vector_db_connector.query_collection(query_text = query,n_results = max_results_query, where = or_condition)

    # print(results)
    distances = [match['score'] for match in results['matches']]
    ids = [match['id'] for match in results['matches']]
    # distances = results['score'][0]
    # ids = results['ids'][0]

    # print(f'{len(ids)} were retrieved!! Before**')

    for prompt in gen_prompts:
        results = vector_db_connector.query_collection(query_text = prompt,n_results = max_results_query, where = or_condition)

        # distances_i = results['distances'][0]
        # ids_i = results['ids'][0]
        distances_i = [match['score'] for match in results['matches']]
        ids_i = [match['id'] for match in results['matches']]
        ids.extend(ids_i)
        distances.extend(distances_i)

    # print(f'Distances: {sorted(distances,reverse=True)}')
    
    # print(f'{len(ids)} were retrieved!!')

    # Remove duplicates by converting the list to a set
    unique_list = list(set(ids))

    # Count how many unique values there are
    num_unique_values = len(unique_list)

    # Display the unique list and count
    # print(f"Number of unique values: {num_unique_values}")

    if len(ids) == 0:

        return None,None,None

    else:

        records = sql_connector.get_records_by_ids(sql_table,ids)

        # Step 1: Convert the records list into a DataFrame
        df = pd.DataFrame(records)
        # print(f'DF: {df}')
        # Step 2: Unpack the 'metadata' column, which is a JSON string, into separate columns
        df['metadata'] = df['metadata'].str.replace("'", '"')
        df_metadata = df['metadata'].apply(json.loads).apply(pd.Series)
        df_metadata.drop(columns=['id'], inplace=True)

        # Step 3: Concatenate the original DataFrame with the unpacked metadata columns
        df_combined = pd.concat([df.drop(columns=['metadata']), df_metadata], axis=1)

        df_distances = pd.DataFrame({'id': ids, 'distance': distances})

        df_merged = pd.merge(df_combined, df_distances, on='id', how='left')

        # Group by 'id' and get the row with the minimum distance for each group
        df_filtered = df_merged.loc[df_merged.groupby('id')['distance'].idxmin()]

        # Optionally, reset the index
        df_filtered = df_filtered.reset_index(drop=True)
        # Filter the DataFrame
        # Filter based on the distance threshold
        df_filtered = df_filtered[df_filtered['distance'] >= distance_threshold]

        # Ensure that df_filtered has at most max_results items, sorted by smallest distance
        df_filtered = df_filtered.sort_values(by='distance',ascending=True).head(max_results)

        filtered_ids = df_filtered['id'].tolist()
        # Convert the filtered DataFrame to a list of dictionaries
        filtered_data = df_filtered.to_dict(orient='records')

        filtered_data = get_ordered_records(filtered_data)
        print(f'filtered_ids {filtered_ids}')

        if len(filtered_data)>0:
            # print(f'Filtered data: {filtered_data}')
            return filtered_data,ids,filtered_ids
        else:
            # print('No data founded')
            return None,ids,None

"""
# Updated retrieve_context function COPY RERANKER
def retrieve_context(chroma_connector, sql_connector, sql_table, query, max_results, retrieve_date_documents, max_results_query, distance_threshold_query, distance_threshold=0.5):

    or_condition = {"$or": []}
    for i in retrieve_date_documents:
        doc = i['type']
        dates = i['dates']
        if len(dates) > 0:
            type_key = i['type_key']
            temp_and = {"$and": [{"valid_date": {'$in': dates}}, {"type_key": {'$eq': type_key}}]}
            if len(retrieve_date_documents) > 1:
                or_condition["$or"].append(temp_and)
            else:
                or_condition = temp_and

    # Query the database (ChromaDB) for documents and their distances
    results = chroma_connector.col.query(query_texts=query, n_results=max_results_query, include=['distances'], where=or_condition)

    distances = results['distances'][0]
    ids = results['ids'][0]

    if len(ids) == 0:
        return None, None, None
    else:
        records = sql_connector.get_records_by_ids(sql_table, ids)

        # Convert the records list into a DataFrame
        df = pd.DataFrame(records)

        # Unpack the 'metadata' column, which is a JSON string, into separate columns
        df_metadata = df['metadata'].apply(json.loads).apply(pd.Series)
        df_metadata.drop(columns=['id'], inplace=True)

        # Concatenate the original DataFrame with the unpacked metadata columns
        df_combined = pd.concat([df.drop(columns=['metadata']), df_metadata], axis=1)

        # Merge distances and records into a single DataFrame
        df_distances = pd.DataFrame({'id': ids, 'distance': distances})
        df_merged = pd.merge(df_combined, df_distances, on='id', how='left')

        # Apply distance threshold filtering
        df_filtered = df_merged[df_merged['distance'] <= distance_threshold]

        # Convert the filtered DataFrame to a list of dictionaries
        filtered_data = df_filtered.to_dict(orient='records')

        # Sort filtered data by IDs using get_ordered_records
        filtered_data = get_ordered_records(filtered_data)

        # Apply re-ranking based on the query and limit the results by max_results
        reranked_data, filtered_out_count = rerank_documents_by_distance_and_model(filtered_data, query, max_results)

        # print(reranked_data)

        if len(reranked_data) > 0:
            print(f"Filtered out {filtered_out_count} documents after re-ranking.")
            return reranked_data, ids, [doc['id'] for doc in reranked_data]
        else:
            return None, ids, None
"""

def get_ordered_records(records):
    # Sort the records based on the id
    return sorted(records, key=lambda x: tuple(map(int, x['id'].split('.'))))

def ids_to_retrieve(records,dif=1):
    context_ranges = []

    # Iterate through the list and process each entry
    for index, entry in enumerate(records):
        entry_doc, entry_chunk = entry['id'].split('.')
        pre_chunk = entry['pre_id'].split('.')[1] if entry['pre_id'] else entry_chunk
        post_chunk = entry['post_id'].split('.')[1] if entry['post_id'] else entry_chunk

        entry_range = {
            'doc': entry_doc,
            'start': int(pre_chunk),
            'end': int(post_chunk)
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
                            break
                    else:
                        if entry_range['end']+dif >= j['start']:
                            # print(f'merge {j} {entry_range}')
                            j['start'] = entry_range['start']
                            break
                
                if n == len(context_ranges)-1:
                    context_ranges.append(entry_range)
                    break

    # Generate the list of IDs for each range
    ranges = []
    for entry_range in context_ranges:
        doc = entry_range['doc']
        start = entry_range['start']
        end = entry_range['end']

        ids = [f"{doc}.{i}" for i in range(start, end + 1)]
        ranges.append(ids)

    return ranges


def build_context(context,sql_connector,table_name):

    if context == None:
        return context

    else:
        # print(f'pre ids {context}')
        ids = ids_to_retrieve(context)
        # print(f'building context: {ids}')

        records = []
        for i in ids:
            # print(f'i: {i}')
            results = sql_connector.get_records_by_ids(table_name, i)
            records.append(results)

        context = []
        table_n = 1
        for record in records:

            context_i = {'id':'',
                        'source':'',
                        'pages':'',
                        'text':'',
                        'paths':[]}
            act_page = None
            for index,sub_rec in enumerate(record):
                metadata = json.loads(sub_rec['metadata'].replace("'",'"'))
                if metadata['type'] == 'Text':
                    text = sub_rec['content']
                else:
                    if metadata['path'] not in context_i['paths']:
                        context_i['paths'].append(metadata['path'])
                        text = f'\nTable {table_n}\n'
                        table_n+=1
                    else:
                        text = ''
                title = metadata['source'].replace('.pdf','')
                if index == 0:
                    context_i['id'] = metadata['post_id']
                    context_i['source'] = title
                    act_page = metadata['page']
                    context_i['pages']=str(act_page)
                    context_i['text'] = f'{text}'

                elif index < len(record)-1:
                    context_i['text'] += f'{text}'
                else:
                    context_i['text'] += f'{text}'
                    if metadata['page'] > act_page:
                        context_i['pages'] = f'{act_page}-{metadata['page']}'
                
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
    



