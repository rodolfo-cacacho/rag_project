import os
import openai
import json
from collections import defaultdict
import random
from utils.chunking_embedding import process_metadata_csv,merge_pages
import random
from pydantic import BaseModel
import math
import base64
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set the seed
seed_value = 15
random.seed(seed_value)
load_dotenv()
API_KEY_CGPT = os.getenv('API_KEY_CGPT')
openai.api_key = API_KEY_CGPT


def create_sample(sql_con,table_chunks_table,
                  eval_chunks_table,eval_chunks_table_schema,
                  metadata_file_path,N=300,overwrite = False):
    
    def calculate_sample_size(total_chunks):
        """Calculate sample size based on document size."""
        if total_chunks <= 10:
            return total_chunks
        return math.floor((total_chunks - 10) / 10) + 10

    chunks = sql_con.get_all_records_as_dict(table_chunks_table)
    metadata_df = process_metadata_csv(metadata_file_path)

    chunks_recent = []
    for chunk in chunks:
        chunk['metadata'] = json.loads(chunk['metadata'])
        chunk['doc_type'] = chunk['metadata']['doc_type']
        chunk['source'] = chunk['metadata']['source']
        chunk['last_id'] = chunk['metadata']['last_id']
        chunk['type'] = chunk['metadata']['type']
        recent = metadata_df[metadata_df['file'] == chunk['source']]['most_recent'].iloc[0]
        chunk['metadata'] = json.dumps(chunk['metadata'],ensure_ascii=False)

        if recent:
            chunks_recent.append(chunk)

    # Group chunks by doc_type
    chunks_by_type = defaultdict(list)
    for chunk in chunks_recent:
        chunks_by_type[chunk['doc_type']].append(chunk)

    sampled_chunks_by_type = {}

    for doc_type, chunks in chunks_by_type.items():
        total_chunks = len(chunks)
        sample_size = calculate_sample_size(total_chunks)
        if total_chunks <= 10:
            # Include all chunks for small documents
            sampled_chunks_by_type[doc_type] = chunks
        else:
            # Random sampling for larger documents
            sampled_chunks_by_type[doc_type] = random.sample(chunks, sample_size)

    for doc_type, sampled_chunks in sampled_chunks_by_type.items():
        for chunk in sampled_chunks:
            # Parse the current chunk's ID
            doc_id, chunk_id = map(int, chunk['id'].split('.'))

            # Determine the previous and next chunk IDs
            prev_chunk_id = chunk_id - 1 if chunk_id > 0 else None
            next_chunk_id = chunk_id + 1

            # Retrieve the last chunk ID for this document from the metadata
            _,last_chunk_id = map(int, chunk['last_id'].split('.'))

            content_merge = []
            # Check and get the previous chunk
            if prev_chunk_id is not None:
                prev_chunk_key = f"{doc_id}.{prev_chunk_id}"
                prev_chunk = next((c for c in chunks_recent if c['id'] == prev_chunk_key), None)
                if prev_chunk:
                    chunk_prev_100_chars = (prev_chunk['content'][-N:],prev_chunk['type'])
                    content_merge.append(chunk_prev_100_chars)

            content_merge.append((chunk['content'],chunk['type']))

            # Check and get the next chunk
            if next_chunk_id <= last_chunk_id:
                next_chunk_key = f"{doc_id}.{next_chunk_id}"
                next_chunk = next((c for c in chunks_recent if c['id'] == next_chunk_key), None)
                if next_chunk:
                    chunk_post_100_chars = (next_chunk['content'][:N],next_chunk['type'])
                    content_merge.append(chunk_post_100_chars)
                   

            if len(content_merge)>1:
                chunk['merged_content'] = merge_pages(content_merge)
            else:
                chunk['merged_content'] = chunk['content']
                 
    sql_con.create_table(eval_chunks_table,eval_chunks_table_schema)

    for i,list_sample in sampled_chunks_by_type.items():

        for chunk in list_sample:
            del chunk["last_id"]
            del chunk["type"]

        sql_con.insert_many_records(eval_chunks_table,list_sample,overwrite = overwrite)

class Question(BaseModel):
    question: str
    type: str

class genQuestions(BaseModel):
    questions: list[Question]

class answerFormat(BaseModel):
    answer: str


def generate_questions(sql_con, table_eval_chunks, table_QAs, table_QAs_schema,table_summaries,overwrite = False):
    """
    Generate questions based on chunk content and insert them into the database.
    
    Args:
        sql_con: SQL connector instance.
        table_eval_chunks: Name of the table containing evaluation chunks.
        table_QAs: Name of the table to store generated questions.
        table_QAs_schema: Schema of the table for storing QAs.
        openai_api_key: OpenAI API key for accessing the GPT model.
    """
    ret_cols = ['doc_type', 'summary', 'summary_revised']

    # Get all evaluation chunks from the database
    chunks = sql_con.get_all_records_as_dict(table_eval_chunks)

    # Prepare the table for storing questions
    sql_con.create_table(table_QAs, table_QAs_schema)

    questions_gen = sql_con.get_all_records_as_dict(table_QAs)

    #  Find unique id_sample values in the questions
    used_id_samples = {q["id_sample"] for q in questions_gen}

    # Filter out chunks that have already been used
    chunks = [chunk for chunk in chunks if chunk["id_sample"] not in used_id_samples]

    # Initialize the progress bar
    with tqdm(total=len(chunks), desc="Generating Questions", unit="chunk") as pbar:
        for chunk in chunks:
            # Extract content and metadata
            content = chunk.get("merged_content", chunk["content"])
            chunk_id = chunk["id_sample"]
            doc_type = chunk["doc_type"]
            text_type = chunk.get("type", "Text")  # Defaults to "Text" if type is not provided
            metadata = json.loads(chunk['metadata'])
            paths = metadata['path']

            retrieve_doc_type = {'doc_type': doc_type}
            doc_type, summary, summary_revised = sql_con.get_record(
                table_summaries, ret_cols, retrieve_doc_type
            ) 

            if paths in ("", []):
                img_paths = None
            else:
                img_paths = paths

            # Instructions for question generation
            instructions = """You are an AI assistant specializing in technical and regulatory topics related to building efficiency and funding in Germany. Your task is to analyze the provided context and generate up to 5 diverse, relevant, and specific questions in German.

1. Use the document summary to understand the purpose, scope, and usage of the document.
2. Ensure all questions are directly tied to the provided context, avoiding ambiguity.
   - Questions must reference the specific information in the context.
   - Avoid broad or unspecified questions that could apply to other parts of the document or dataset.
3. Avoid irrelevant or trivial questions, such as:
   - Questions about document versions or publication dates.
   - Questions that are not specific to the provided context.
4. Focus on generating questions that reflect how this document can be used by experts for regulatory, technical, or planning purposes.
5. Ensure each question is unique, clear, and adds value to understanding the content."""


            # Prompt for GPT
            prompt = f"""Your task is to generate up to 5 questions based on the provided context and summary of the document. Follow these rules:

1. Questions must be in **German** and fully answerable using the provided context.
2. Use the document summary as a guide to align the questions with the purpose, scope, and usage of the document.
3. Avoid ambiguous or vague questions. Every question must:
   - Be specific to the provided context.
   - Clearly reference the relevant topic or information.
   - Avoid general phrasing that could apply to other sections or documents.
4. Avoid irrelevant or trivial questions, such as:
   - Questions about the document’s version or publication date.
   - Questions that are not directly tied to the provided context.
5. Focus on generating questions that align with the following types:
   - **Factual**: Specific thresholds, limits, or technical details (e.g., "Welche Wärmepumpen erfüllen die technischen Anforderungen der BEG?").
   - **Procedural**: Steps, processes, or documentation requirements (e.g., "Welche Nachweise sind erforderlich, um Fördermittel zu beantragen?").
   - **Analytical**: Implications, reasoning, or relationships between requirements (e.g., "Warum ist die Wahl des Kältemittels relevant für die Förderung?").
6. Ensure the number of questions reflects the richness of the content. Fewer questions are acceptable if the content is limited.

Document Type: {doc_type}
Summary: {summary}

Context:
{content}"""
            # Call GPT API
            answer = call_gpt_api_with_multiple_images(
                instructions=instructions,
                prompt=prompt,
                response_format=genQuestions,
                img_paths=img_paths
            )

            # Parse and process the response
            answer = json.loads(answer)
            questions = answer['questions']

            # Insert questions into the database
            for question in questions:
                sql_con.insert_many_records(table_QAs, [{
                    "id_sample": chunk_id,
                    "question": question['question'],
                    "type_question": question['type'],
                    "type_content": text_type
                }], overwrite=overwrite)

            # Update the progress bar
            pbar.update(1)

    print("Questions generation completed.")



def generate_answers(sql_con, table_eval_chunks, table_QAs):
    """
    Generate answers based on chunk content and insert them into the database.

    Args:
        sql_con: SQL connector instance.
        table_eval_chunks: Name of the table containing evaluation chunks.
        table_QAs: Name of the table to store generated questions.
    """
    ret_cols = ['doc_type', 'metadata', 'merged_content']
    # Get all evaluation chunks from the database
    questions = sql_con.get_all_records_as_dict(table_QAs)

    questions = [q for q in questions if q['expected_answer'] is None]

    # Initialize the progress bar
    with tqdm(total=len(questions), desc="Generating Answers", unit="answer") as pbar:
        for qa in questions:
            question = qa["question"]
            id_q = qa['id_question']
            id_sample = qa["id_sample"]
            retrieve_id = {'id_sample': id_sample}
            doc_type, metadata, merged_content = sql_con.get_record(
                table_eval_chunks, ret_cols, retrieve_id
            )  # Get the associated chunk
            metadata = json.loads(metadata)
            paths = metadata['path']

            if paths in ("", []):
                img_paths = None
            else:
                img_paths = paths

            content = merged_content

            instructions = """You are an AI assistant with expertise in technical and regulatory topics related to building efficiency and funding in Germany. Your task is to generate an expected answer for a given question based on the provided context. This expected answer will act as a guide for evaluating actual answers.

1. The expected answer must:
   - Be precise, concise, and directly relevant to the question.
   - Fully cover all key points required to answer the question, based solely on the provided context.
   - Avoid including information not found in the context, even if it might be generally correct.

2. If the context cannot fully answer the question, explicitly mention what is missing or ambiguous.

3. Structure the expected answer as a **guide**, listing key elements that must be covered.

4. The output must be in **German** and free of ambiguity."""

            # Prepare the prompt
            prompt = f"""Your task is to generate an expected answer for the provided question. This expected answer will act as a guide for evaluating actual answers. Follow these rules:

1. Base the answer strictly on the provided context. Do not include information outside the context.
2. The expected answer must:
   - Be precise and concise.
   - Fully address the question, covering all relevant points from the context.
   - Be clear and specific, avoiding ambiguous or overly broad statements.
3. If the context does not fully address the question, explicitly state what is missing.

### Information Provided:
- **Document**: {doc_type}
- **Context**:
{content}

- **Question**: 
{question}

Generate the expected answer in **German**."""

            answer = call_gpt_api_with_multiple_images(
                instructions=instructions,
                prompt=prompt,
                response_format=answerFormat,
                img_paths=img_paths,
                max_tokens=4000
            )
            answer = json.loads(answer)
            answer_q = answer['answer']

            # Update the QA table with the answer
            upd_data = {'expected_answer': answer_q}
            condition = {'id_question': id_q}
            sql_con.update_record(table_QAs, upd_data, condition)

            # Update the progress bar
            pbar.update(1)

    print("Answer generation completed.")


def call_gpt_api_with_multiple_images(instructions, prompt, model="gpt-4o-2024-08-06", max_tokens=2500, response_format=None, img_paths=None, detail='high'):
    """
    Sends a single message to GPT API with optional multiple image inputs and retrieves the response.
    
    Parameters:
    - instructions: System instructions to set the context (e.g., "You are an AI assistant that analyzes tables").
    - prompt: User's message or query (e.g., "Please analyze the tables in the images and provide a summary").
    - model: The GPT model to be used (default is "gpt-4o-2024-08-06").
    - max_tokens: Maximum number of tokens for the response (default is 2500).
    - response_format: Format of the response (e.g., "Rag_reponse"). Defaults to standard completion if not provided.
    - img_paths: Optional list of image file paths. If provided, images will be included in the request.
    - detail: The detail level for image analysis, if applicable.
    
    Returns:
    - The GPT answer object.
    """

    content = []
    dict_images = []

    # Create the messages list to send to GPT
    messages = [
        {"role": "system", "content": instructions}
    ]

    # Process image paths if provided
    if img_paths:
        for img_path in img_paths:
            try:
                # Encode the image in base64
                base64_image = encode_image(img_path)
                dic_images = {
                    'type': 'image_url',
                    'image_url': {'url': f"data:image/png;base64,{base64_image}", 'detail': detail}
                }
                dict_images.append(dic_images)
            except Exception as e:
                print(f"Error encoding image {img_path}: {e}")
                continue

    # Combine prompt and image information
    prompt_text = {'type': 'text', 'text': prompt}
    content.append(prompt_text)

    # Add image data if available
    if dict_images:
        content.extend(dict_images)

    # Add the content to the user message
    messages.append({"role": "user", "content": content})

    try:
        if response_format is None:
            # Call GPT API without a specific response format
            response = openai.beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
        else:
            # Call GPT API with a specified response format
            response = openai.beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format
            )

        # Extract and return the response content
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        print(f"Error during GPT API call: {e}")
        return None

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

def evaluate_q_similarity(db, table_name, embedding_handler, threshold=0.95):
    """
    Evaluate the similarity of questions in the database and update the similarity column.

    Args:
        db (MySQLDB): An instance of the MySQLDB class for database operations.
        table_name (str): The name of the table containing the questions.
        embedding_handler (EmbeddingHandler): An instance of the EmbeddingHandler for embedding text.
        threshold (float): The similarity threshold for grouping questions.
    """
    # Step 1: Retrieve all questions from the database
    questions_data = db.get_all_records_as_dict(table_name)
    if not questions_data:
        print("No questions found in the database.")
        return

    # Extract the questions and their IDs
    question_ids = [row['id_question'] for row in questions_data]
    questions = [row['question'] for row in questions_data]

    # Step 2: Embed the questions
    embeddings = np.array(embedding_handler.embed_texts(questions))

    # Step 3: Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Step 4: Group similar questions based on the threshold
    similarity_results = {}
    for i, row in enumerate(similarity_matrix):
        # Pair similarity scores with question IDs
        similar_pairs = [
            (question_ids[j], sim) for j, sim in enumerate(row) if sim > threshold and i != j
        ]
        # Sort by similarity score in descending order
        similar_pairs = sorted(similar_pairs, key=lambda x: x[1], reverse=True)
        # Store only question IDs, excluding self
        similar_question_ids = [qid for qid, _ in similar_pairs]
        similarity_results[question_ids[i]] = similar_question_ids

    # Step 5: Update the database with the similarity results
    for question_id, similar_ids in similarity_results.items():
        db.update_record(
            table_name,
            update_data={'sim': ",".join(map(str, similar_ids))},
            conditions={'id_question': question_id},
            append=True  # Ensure that updates can append to the field if needed
        )
    print("Similarity evaluation and updates are complete.")

class askIDsFormat(BaseModel):
    worth_asking: list[int]



def llm_eval_similarity(sql_con, qas_table):
    """
    Prepare prompts for evaluating the most worth-asking questions from similar question groups.
    Tracks pairwise comparisons to avoid redundant evaluations and updates the database.

    Args:
        sql_con: Database connection object.
        qas_table: Name of the questions table in the database.
    """
    # Step 1: Retrieve all questions from the database
    questions_eval = sql_con.get_all_records_as_dict(qas_table)

    # Filter only the questions with similarity groups
    questions_sim = [question for question in questions_eval if question['sim']]

    # Step 2: Initialize comparison tracking dictionary
    comparison_dict = {q['id_question']: set() for q in questions_eval}

    # Initialize progress bar
    progress_bar = tqdm(total=len(questions_sim), desc="Processing Questions")

    # Prepare prompts for each group
    for question_data in questions_sim:
        question_id = question_data['id_question']
        similar_ids = question_data['sim'].split(",")
        similar_ids = [int(sim_id) for sim_id in similar_ids]

        # Filter out already compared questions
        unprocessed_ids = [
            sim_id for sim_id in similar_ids if sim_id not in comparison_dict[question_id]
        ]

        # Skip if no new comparisons to process
        if not unprocessed_ids:
            progress_bar.update(1)
            continue

        # Collect the main question and its unprocessed similar ones
        main_question = f"Question {question_id}: '{question_data['question']}'"
        similar_questions = [
            f"Question {sim_id}: '{questions_eval[sim_id - 1]['question']}'"  # Adjust indexing as needed
            for sim_id in unprocessed_ids
        ]

        # Mark these questions as compared for both the main and similar questions
        for sim_id in unprocessed_ids:
            comparison_dict[question_id].add(sim_id)
            comparison_dict[sim_id].add(question_id)

        instructions = """You are an AI assistant with expertise in technical and regulatory topics related to building efficiency and funding in Germany.
        You are tasked with evaluating a group of questions to determine which are worth asking. Use the following criteria to make your decisions:
        1. Clarity:
        - Is the question clearly phrased and easy to understand?
        - Avoid vague, ambiguous, or overly complex questions.
        2. Specificity:
        - Does the question require detailed, precise, and focused answers?
        - Avoid overly broad or generic questions.
        3. Relevance:
        - Does the question align with the context or task it pertains to?
        - Ensure the question matches the intended dataset or domain.
        4. Uniqueness:
        - Does the question provide unique value compared to others in the group?
        - Avoid redundant questions that seek similar or overlapping information."""

        # Construct the prompt
        prompt_similitude = f"""
        You are given a group of similar questions. Evaluate which questions are the most worth asking based on clarity, specificity, and relevance to the context. Return the list with the questions worth asking.

        Here are the questions:
        {main_question}
        {chr(10).join(similar_questions)}

        Choose the top question and any other significantly different and worth asking. Avoid questions deemed too similar.
        """

        # Call the GPT API to get worth-asking question IDs
        response = call_gpt_api_with_multiple_images(
            instructions=instructions,
            prompt=prompt_similitude,
            max_tokens=500,
            response_format=askIDsFormat
        )

        # Parse and process the response
        answer = json.loads(response)
        ids_worth_asking = answer['worth_asking']

        # Update the database for the worth-asking questions
        for qid in ids_worth_asking:
            sql_con.update_record(
                table_name=qas_table,
                update_data={'sim_worth': True},  # Set the flag to True
                conditions={'id_question': qid}
            )

        # Debug print the processed IDs
        # print(f"Processed Question ID {question_id}, Worth Asking IDs: {ids_worth_asking}")

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()
    print(f"All prompts have been processed. Database updated with worth-asking questions.")

class scoreQuestion(BaseModel):
    clarity: int
    specificity: int
    relevance: int

def llm_eval_question_answers(sql_con, qas_table):
    """
    Evaluate questions using clarity, specificity, and relevance criteria, and update scores in the database.
    Filters out questions with sim != "" and sim_worth not equal to 1 or True.

    Args:
        sql_con: Database connection object.
        qas_table: Name of the questions table in the database.
    """
    # Step 1: Retrieve all questions from the database
    questions_eval = sql_con.get_all_records_as_dict(qas_table)

    # Step 2: Filter questions
    filtered_questions = [
        question for question in questions_eval
        if not (question['sim'] and not question['sim_worth'])
    ]

    # Initialize progress bar
    progress_bar = tqdm(total=len(filtered_questions), desc="Evaluating Questions")

    for question_data in filtered_questions:
        question_id = question_data['id_question']
        question_text = question_data['question']
        expected_answer = question_data.get('expected_answer', "No answer provided.")

        # Construct the prompt for the LLM
        prompt_eval = f"""Evaluate the following question based on the criteria below and provide a score (1-5) for each:
        1. Clarity:
        - Is the question clearly phrased and easy to understand?
        - Avoid vague, ambiguous, or overly complex questions.
        2. Specificity:
        - Does the question require detailed, precise, and focused answers?
        - Avoid overly broad or generic questions.
        3. Relevance:
        - Does the question align with the context or task it pertains to?
        - Ensure the question matches the intended dataset or domain.

        Question: {question_text}

        Expected Answer: {expected_answer}
        """

        # Call GPT API to get scores
        response = call_gpt_api_with_multiple_images(
            instructions="You are an AI assistant with expertise in technical and regulatory topics related to building efficiency and funding in Germany. Evaluate the question on clarity, relevance, and specificity.",
            prompt=prompt_eval,
            max_tokens=500,
            response_format=scoreQuestion
        )

        # Parse the response
        answer = json.loads(response)
        clarity = answer['clarity']
        specificity = answer['specificity']
        relevance = answer['relevance']

        # Update the database with the scores
        sql_con.update_record(
            table_name=qas_table,
            update_data={
                'clarity': clarity,
                'specificity': specificity,
                'relevance': relevance
            },
            conditions={'id_question': question_id}
        )

        # Debug print the processed question and scores (optional)
        # print(f"Processed Question ID {question_id}, Scores: Clarity={clarity}, Specificity={specificity}, Relevance={relevance}")

        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print(f"All {len(filtered_questions)} questions have been evaluated and updated.")

def llm_eval_question(sql_con, qas_table):
    """
    Evaluate questions using clarity, specificity, and relevance criteria, and update scores in the database.
    Filters out questions with sim != "" and sim_worth not equal to 1 or True.

    Args:
        sql_con: Database connection object.
        qas_table: Name of the questions table in the database.
    """
    # Step 1: Retrieve all questions from the database
    questions_eval = sql_con.get_all_records_as_dict(qas_table)

    # Step 2: Filter questions
    filtered_questions = [
        question for question in questions_eval
        if not (question['sim'] and not question['sim_worth'])
    ]

    # Initialize progress bar
    progress_bar = tqdm(total=len(filtered_questions), desc="Evaluating Questions")

    for question_data in filtered_questions:
        question_id = question_data['id_question']
        question_text = question_data['question']
        # expected_answer = question_data.get('expected_answer', "No answer provided.")

        # Construct the prompt for the LLM
        prompt_eval = f"""Evaluate the following question based on the criteria below and provide a score (1-5) for each:
        1. Clarity:
        - Is the question clearly phrased and easy to understand?
        - Avoid vague, ambiguous, or overly complex questions.
        2. Specificity:
        - Does the question require detailed, precise, and focused answers?
        - Avoid overly broad or generic questions.
        3. Relevance:
        - Does the question align with the context or task it pertains to?
        - Ensure the question matches the intended dataset or domain.

        Question: {question_text}
        """

        # Call GPT API to get scores
        response = call_gpt_api_with_multiple_images(
            instructions="You are an AI assistant with expertise in technical and regulatory topics related to building efficiency and funding in Germany. Evaluate the question on clarity, relevance, and specificity. The questions are related to the BEG program, addressing different aspects of funding, eligibility, and technical standards.",
            prompt=prompt_eval,
            max_tokens=500,
            response_format=scoreQuestion
        )

        # Parse the response
        answer = json.loads(response)
        clarity = answer['clarity']
        specificity = answer['specificity']
        relevance = answer['relevance']

        # Update the database with the scores
        sql_con.update_record(
            table_name=qas_table,
            update_data={
                'clarity_q': clarity,
                'specificity_q': specificity,
                'relevance_q': relevance
            },
            conditions={'id_question': question_id}
        )

        # Debug print the processed question and scores (optional)
        # print(f"Processed Question ID {question_id}, Scores: Clarity={clarity}, Specificity={specificity}, Relevance={relevance}")

        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print(f"All {len(filtered_questions)} questions have been evaluated and updated.")

def select_valid_questions(sql_con, qas_table):
    """
    Select valid questions based on score criteria and mark them as valid in the database.

    Args:
        sql_con: Database connection object.
        qas_table: Name of the questions table in the database.
    """
    # Step 1: Retrieve all questions from the database
    questions = sql_con.get_all_records_as_dict(qas_table)

    # Initialize a list to track valid question IDs
    valid_question_ids = []

    for question in questions:
        question_id = question['id_question']

        # Handle None values by treating them as 0
        clarity = question.get('clarity', 0) or 0
        specificity = question.get('specificity', 0) or 0
        clarity_q = question.get('clarity_q', 0) or 0
        specificity_q = question.get('specificity_q', 0) or 0

        # Step 2: Apply selection criteria
        if (clarity + clarity_q > 8) and (specificity + specificity_q > 8):
            valid_question_ids.append(question_id)

    # Step 3: Update the database for valid questions
    for question_id in valid_question_ids:
        sql_con.update_record(
            table_name=qas_table,
            update_data={'valid': True},
            conditions={'id_question': question_id}
        )

    print(f"{len(valid_question_ids)} questions marked as valid.")


