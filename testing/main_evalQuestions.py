import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (CONFIG_SQL_DB,DB_NAME,
                    SQL_EVAL_QAS_TABLE_SCHEMA,
                    SQL_EVAL_QAS_TABLE, 
                    EMBEDDING_MODEL,EMBEDDING_MODEL_API,
                    EMBEDDING_MODEL_EMB_TASK)
from utils.MySQLDB_manager import MySQLDB
from testing.modules.testing_modules import (evaluate_q_similarity,llm_eval_similarity,
                                             llm_eval_question,llm_eval_question_answers,
                                             select_valid_questions)
from utils.embedding_handler import EmbeddingHandler

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)
sql_con.create_table(SQL_EVAL_QAS_TABLE,SQL_EVAL_QAS_TABLE_SCHEMA)

embed_handler = EmbeddingHandler(model_name=EMBEDDING_MODEL,
                                 use_api=EMBEDDING_MODEL_API,
                                 task=EMBEDDING_MODEL_EMB_TASK
                                 )

evaluate_q_similarity(embedding_handler=embed_handler,
                      table_name= SQL_EVAL_QAS_TABLE,
                      db=sql_con,
                      threshold=0.92)

llm_eval_similarity(sql_con=sql_con,
                    qas_table=SQL_EVAL_QAS_TABLE)

llm_eval_question(sql_con=sql_con,
                  qas_table = SQL_EVAL_QAS_TABLE)
llm_eval_question_answers(sql_con=sql_con,
                  qas_table = SQL_EVAL_QAS_TABLE)

select_valid_questions(sql_con=sql_con,
                       qas_table = SQL_EVAL_QAS_TABLE)

