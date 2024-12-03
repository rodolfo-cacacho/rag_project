import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv

from config import (EMBEDDING_MODELS,SUFFIX,DB_NAME,CONFIG_SQL_DB,TABLE_STORE_DIR,METADATA_FILE_PATH,)
from utils.pinecone_hybrid_connector import PineconeDBConnectorHybrid
from utils.MySQLDB_manager import MySQLDB
from utils.chunking_embedding import semantic_chunking,embedding_bm25_calculation,process_and_upload
from utils.embedding_handler import EmbeddingHandler
import spacy

nlp = spacy.load('de_core_news_lg')

load_dotenv()
API_PINE_CONE = os.getenv('API_PINE_CONE')

MAX_TOKENS = 125
sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

sql_chunks_table = f'chunks_table_{SUFFIX}_{MAX_TOKENS}'
sql_table_vocab = f'vocabulary_bm25_{SUFFIX}_{MAX_TOKENS}'

print("Creating Chunks")

records = semantic_chunking(sql_con=sql_con,
                  table_chunks_name=sql_chunks_table,
                  df_code_path=METADATA_FILE_PATH,
                  max_tokens=MAX_TOKENS,
                  output_dir=TABLE_STORE_DIR)

print("Creating BM25 Vocabulary")
upsert_list = embedding_bm25_calculation(sql_con=sql_con,
                                        table_name=sql_chunks_table,
                                        table_store=sql_table_vocab,
                                        json_path=TABLE_STORE_DIR,
                                        nlp=nlp)

for idx_model,embedding_model in enumerate(EMBEDDING_MODELS):
    
    print(f"Calculating Embeddings: {embedding_model}")
    
    embedding_model_name = embedding_model.split("/")[1].replace('_','-').lower()
    embedding_model_dim = EMBEDDING_MODELS[embedding_model]["dimension"]
    embedding_model_api = EMBEDDING_MODELS[embedding_model]["api_usage"]
    embedding_model_ret_task = EMBEDDING_MODELS[embedding_model]["retrieve_task"]
    embedding_model_emb_task = EMBEDDING_MODELS[embedding_model]["embed_task"]

    index_name = f'{embedding_model_name}-{SUFFIX}-{MAX_TOKENS}'

    vec_con = PineconeDBConnectorHybrid(api_key=API_PINE_CONE,
                                    index_name=index_name,
                                    dimension=embedding_model_dim)
    
    embedding_handler = EmbeddingHandler(
        model_name=embedding_model,  # Replace with your transformer model
        use_api=False,  # Set to True if using API
        task=embedding_model_emb_task
    )

    process_and_upload(chunks=upsert_list,
                   pinecone_connector=vec_con,
                   embedding_handler=embedding_handler,
                   use_sparse=True,
                   batch_size_embedding=16,
                   batch_size_upsert=16
                   )
