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
                    EMBEDDING_MODEL_EMB_TASK,
                    TEST_RESULTS_TABLE,SQL_EVAL_CHUNKS_TABLE,
                    SQL_PROMPTS_TABLE)
from utils.MySQLDB_manager import MySQLDB
from testing.modules.evaluating_modules import RAGEvaluator

TEST_NAME = "testJinaV3500"

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

ragEval = RAGEvaluator(sql_con=sql_con,
                       test_name=TEST_NAME,
                       test_table_name=TEST_RESULTS_TABLE,
                       qas_table_name=SQL_EVAL_QAS_TABLE,
                       chunks_eval_table_name=SQL_EVAL_CHUNKS_TABLE,
                       prompts_table_name=SQL_PROMPTS_TABLE)

df_results = ragEval.data_df


print(ragEval.generate_report())
