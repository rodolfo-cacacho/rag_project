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


def generate_questions(sql_con, table_eval_chunks, table_QAs, table_QAs_schema,overwrite = False):
    """
    Generate questions based on chunk content and insert them into the database.
    
    Args:
        sql_con: SQL connector instance.
        table_eval_chunks: Name of the table containing evaluation chunks.
        table_QAs: Name of the table to store generated questions.
        table_QAs_schema: Schema of the table for storing QAs.
        openai_api_key: OpenAI API key for accessing the GPT model.
    """

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

            if paths in ("", []):
                img_paths = None
            else:
                img_paths = paths

            # Instructions for question generation
            instructions = (
                "You are an AI assistant with expertise in technical and regulatory topics related to building "
                "efficiency and funding in Germany. Your task is to analyze the provided context and generate up to "
                "5 questions in German. Ensure the questions are diverse, relevant, and directly answerable from the "
                "context. Focus on technical requirements, funding criteria, and procedural details, etc. Avoid "
                "repetition and ensure clarity in questions."
            )

            # Prompt for GPT
            prompt = f"""Your task is to generate questions from the given context. Follow these instructions:
1. Generate up to 5 questions in **German**. Every question should be different and not only use different wording. The number of questions should depend on the richness of the content.
2. Each question must be fully answerable from the given context.
3. The answers should not contain any links.
4. The questions should be of moderate difficulty.
5. The question must be reasonable and must be understood and responded to by humans.
6. Do not use phrases like 'provided context', etc., in the question.
7. Questions can be of types:
   - Factual: Thresholds, limits, or details.
   - Procedural: Steps or processes.
   - Analytical: Implications or reasoning.
8. The questions must be in **German**.

Document: {doc_type}

Context: {content}
"""
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

            instructions = "You are an AI assistant with expertise in technical and regulatory topics related to building efficiency and funding in Germany. Your task is to analyze the provided context and answer the question in German. Ensure the answer is precise and concise."

            # Prepare the prompt
            prompt = f"""Answer the following question based on the provided context. Be precise and concise. Answer in **German**.
            
            Document: {doc_type}

            Context:
            {content}
            
            Question:
            {question}
            """

            answer = call_gpt_api_with_multiple_images(
                instructions=instructions,
                prompt=prompt,
                response_format=answerFormat,
                img_paths=img_paths,
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

