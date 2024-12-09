# RAG CONFIGURATIONS
MAX_TOKENS = 500
SUFFIX = 'clean'

EMBEDDING_MODELS = {
    "jinaai/jina-embeddings-v2-base-de":{"dimension":768,
                                         "embed_task":None,
                                         "retrieve_task":None,
                                         "api_usage":False,
                                         "instruction":None,
                                         "normalize":False},
    "jinaai/jina-embeddings-v3":{"dimension":1024,
                                 "embed_task":"retrieval.passage",
                                 "retrieve_task":"retrieval.query",
                                 "api_usage":False,
                                 "instruction":None,
                                 "normalize":False},
    "aari1995/German_Semantic_V3":{"dimension":1024,
                                   "embed_task":None,
                                   "retrieve_task":None,
                                   "api_usage":False,
                                   "instruction":None,
                                   "normalize":False},
    "intfloat/multilingual-e5-large-instruct":{"dimension":1024,
                                   "embed_task":None,
                                   "retrieve_task":None,
                                   "api_usage":False,
                                   "instruction":"Given a query, retrieve relevant information from the available documents",
                                   "normalize":True}}

EMBEDDING_MODEL = "aari1995/German_Semantic_V3"
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL.split("/")[1].replace('_','-').lower()
EMBEDDING_MODEL_DIM = EMBEDDING_MODELS[EMBEDDING_MODEL]["dimension"]
EMBEDDING_MODEL_API = EMBEDDING_MODELS[EMBEDDING_MODEL]["api_usage"]
EMBEDDING_MODEL_RET_TASK = EMBEDDING_MODELS[EMBEDDING_MODEL]["retrieve_task"]
EMBEDDING_MODEL_EMB_TASK = EMBEDDING_MODELS[EMBEDDING_MODEL]["embed_task"]
EMBEDDING_MODEL_INSTRUCTION = EMBEDDING_MODELS[EMBEDDING_MODEL]["instruction"]
EMBEDDING_MODEL_NORMALIZE = EMBEDDING_MODELS[EMBEDDING_MODEL]["normalize"]
 
INDEX_NAME = f'{EMBEDDING_MODEL_NAME}-{SUFFIX}-{MAX_TOKENS}'

MX_RESULTS = 5
DIST_THRESHOLD = 0.2

MX_RESULTS_QUERY = 25

ALPHA_VALUE = 0.9

GEN_PROMPTS = 3

# Directory & File CONFIG

METADATA_DIR = 'data/documents/metadata'

RESULTS_DIR = 'results/'

TABLE_STORE_DIR = 'data/storage/tables'

METADATA_FILE_PATH = f'{METADATA_DIR}/Files_date_version.csv'

USERS_SERVERS = ['root','ubuntu']

TG_SESSION_PATH = 'testing/sessions'

# Bot CONFIG

BOT_NAME = 'THWS Bau Bot'
BBOT_USER = '@BundesBau_bot'
THWS_BOT_USER = '@ThwsBauBot'
BOT_TEST_USER = '@ragtesting'
NOTIFY_USER = '@rodolfocco'
ADMIN_USER = '@rodolfoccr'
TESTING_USERS = ['ragtesting','rodolfocco']

# MySQL CONFIG
CONFIG_SQL_DB = {
    'user': 'root',
    'password': 'admin123',
    'host': 'localhost'
}

DB_NAME = 'data_rag'

# SQL TABLES

SQL_CHUNK_TABLE = f'chunks_table_{SUFFIX}_{MAX_TOKENS}'

SQL_CHUNK_TABLE_SCHEMA = {
    'id': 'varchar(10) NOT NULL PRIMARY KEY',
    'content': 'longtext NOT NULL',
    'metadata': 'longtext NOT NULL'
}

SQL_VOCAB_BM25_TABLE = f'vocabulary_bm25_{SUFFIX}_{MAX_TOKENS}'

SQL_VOCAB_BM25_TABLE_SCHEMA = {
    "id": "INT NOT NULL PRIMARY KEY",
    "word": "VARCHAR(255) NOT NULL",
    "idf": "FLOAT",
    "synonyms": "LONGTEXT"
}

SQL_USER_TABLE = 'user_db'

SQL_USER_TABLE_SCHEMA = {
    'id': 'BIGINT PRIMARY KEY',
    'username': 'VARCHAR(100) NULL',
    'name': 'VARCHAR(100)',
    'date_added': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'date_edit': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP',
    'allowed': 'BOOLEAN DEFAULT FALSE'
}


SQL_MESSAGES_TABLE = 'messages'

SQL_MESSAGES_TABLE_SCHEMA = {
    'auto_message_id': 'BIGINT AUTO_INCREMENT PRIMARY KEY',
    'chat_id': 'BIGINT NOT NULL',
    'message_id': 'BIGINT NOT NULL',
    'role': 'VARCHAR(10)',
    'message': 'longtext NOT NULL',
    'date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'reply_message_id': 'BIGINT NULL',
    'prompt_id': 'BIGINT NULL, FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id) ON DELETE SET NULL',
    'device':'VARCHAR(20), FOREIGN KEY (device) REFERENCES prompts(device) on DELETE SET NULL'
}

SQL_PROMPTS_TABLE = 'prompts'

SQL_PROMPTS_TABLE_SCHEMA = {
    'prompt_id': 'BIGINT AUTO_INCREMENT PRIMARY KEY',
    'device': "VARCHAR(20)",
    'chat_id': 'BIGINT NOT NULL',
    'question': 'longtext NOT NULL',
    'context': 'longtext',
    'question_w_context': 'longtext',
    'answer': 'longtext',
    'evaluation': 'VARCHAR(100)',
    'comment': 'longtext',
    'query_date': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'begin_date': 'TIMESTAMP NOT NULL',
    'end_date': 'TIMESTAMP NOT NULL',
    'completion_tokens': 'INT',
    'prompt_tokens': 'INT',
    'total_tokens': 'INT',
    'context_used': 'LONGTEXT',
    'context_eval': 'varchar(100) DEFAULT NULL',
    'context_ids': 'longtext',
    'context_ids_total': 'longtext',
    'alternate_prompts': 'longtext',
    'embedding_model': 'VARCHAR(100)',
    'chunk_size': 'INT',
    'alpha_value': 'float',
    'improved_query':'longtext',
    'query_intent':'longtext',
    'keyterms':'longtext',
    'setting':'varchar(100)'
}

SQL_EVAL_CHUNKS_TABLE = "eval_chunks"

# SQL query to create the table
SQL_EVAL_CHUNKS_TABLE_SCHEMA = {
    "id_sample": "INT AUTO_INCREMENT PRIMARY KEY",  # Unique record ID
    "id": "VARCHAR(10) NOT NULL",              # Reference to the chunk ID
    "doc_type": "VARCHAR(255) NOT NULL",     # Type of the document
    "source": "VARCHAR(255) NOT NULL",       # Source file or document
    "content": "longtext",
    "merged_content":"longtext",
    "metadata":"longtext",
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"  # Record creation timestamp
}

SQL_EVAL_QAS_TABLE = "eval_QAs"
SQL_EVAL_QAS_TABLE_SCHEMA = {
    "id_question": "INT AUTO_INCREMENT PRIMARY KEY",  # Unique record ID for each QA pair
    "id_sample": f"INT NULL, FOREIGN KEY (id_sample) REFERENCES {SQL_EVAL_CHUNKS_TABLE}(id_sample) ON DELETE SET NULL",                      # Foreign key referencing eval_chunks
    "type_content": "VARCHAR(25)",
    "type_question": "VARCHAR(25)",
    "question": "longtext",                      # Generated question
    "expected_answer": "longtext",                        # Expected answer (nullable)
    "expected_answer_original": "longtext",                        # Expected answer (nullable)
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",  # Record creation timestamp
    "sim": "VARCHAR(500)",
    "sim_worth":"BOOLEAN DEFAULT FALSE",
    "clarity":"int",
    "relevance":"int",
    "specificity":"int",
    "clarity_q":"int",
    "relevance_q":"int",
    "specificity_q":"int",
    "valid":"BOOLEAN DEFAULT FALSE"
}

SQL_DOC_TYPE_SUMMARIES_TABLE = "doc_type_summaries"

SQL_DOC_TYPE_SUMMARIES_TABLE_SCHEMA = {
    'id':'int auto_increment primary key',
    'doc_type':'varchar(255)',
    'source':'varchar(255)',
    'summary':'longtext',
    'summary_revised':'longtext'
}

SQL_DOCUMENTS_TABLE = 'table_documents'

# Constants for table and schema
TEST_RESULTS_TABLE = "test_results"

TEST_RESULTS_SCHEMA = {
    "id": "INT AUTO_INCREMENT PRIMARY KEY",
    "prompt_id": "bigint",
    "device": "VARCHAR(20)",
    "test_name": "VARCHAR(255)",
    "id_question": "INT",
    "status": "ENUM('pending', 'success', 'error') DEFAULT 'pending'",
    "error_message": "TEXT",
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
}

TEST_GEN_ANSWERS_TABLE = "test_gen_answers"

TEST_GEN_ANSWERS_SCHEMA = {
    "id": "INT AUTO_INCREMENT PRIMARY KEY",
    "test_name":"VARCHAR(255)",
    "id_question":"INT",
    "score":"INT",
    "comment":"longtext",
    "precisionA": "float",
    "recallA": "float",
    "f1_scoreA": "float"
}