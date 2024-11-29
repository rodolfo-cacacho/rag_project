import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv

from config import (EMBEDDING_MODELS,SUFFIX,DB_NAME,CONFIG_SQL_DB,TABLE_STORE_DIR)
from utils.hybrid_connector import PineconeDBConnectorHybrid
from utils.MySQLDB_manager import MySQLDB
from utils.chunking_embedding import embedding_bm25_calculation,process_and_upload
from utils.embedding_handler import EmbeddingHandler

load_dotenv()
API_PINE_CONE = os.getenv('API_PINE_CONE')

MAX_TOKENS = 500
sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

sql_chunks_table = f'chunks_table_{SUFFIX}_{MAX_TOKENS}'
sql_table_vocab = f'vocabulary_bm25_{SUFFIX}_{MAX_TOKENS}'

print("Creating BM25 Vocabulary")
upsert_list = embedding_bm25_calculation(sql_con=sql_con,
                                        table_name=sql_chunks_table,
                                        table_store=sql_table_vocab,
                                        json_path=TABLE_STORE_DIR)

for idx_model,embedding_model in enumerate(EMBEDDING_MODELS):
    print(embedding_model)
    embedding_model_name = embedding_model.split("/")[1].replace('_','-').lower()
    embedding_model_dim = EMBEDDING_MODELS[embedding_model]["dimension"]
    embedding_model_api = EMBEDDING_MODELS[embedding_model]["api_usage"]
    embedding_model_ret_task = EMBEDDING_MODELS[embedding_model]["retrieve_task"]
    embedding_model_emb_task = EMBEDDING_MODELS[embedding_model]["embed_task"]

    index_name = f'{embedding_model_name}-{SUFFIX}-{MAX_TOKENS}'

    vec_con = PineconeDBConnectorHybrid(api_key=API_PINE_CONE,
                                    index_name=index_name,
                                    embedding_model_name_dense=embedding_model,
                                    dimension=embedding_model_dim)
    
    embedding_handler = EmbeddingHandler(
        model_name=embedding_model,  # Replace with your transformer model
        use_api=False,  # Set to True if using API
        task=embedding_model_emb_task
    )

    process_and_upload(chunks=upsert_list,
                   pinecone_connector=vec_con,
                   embedding_handler=embedding_handler,
                   use_sparse=True
                   )





