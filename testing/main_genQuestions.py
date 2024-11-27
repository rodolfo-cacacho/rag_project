import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (CONFIG_SQL_DB,DB_NAME,
                    SQL_EVAL_CHUNKS_TABLE,
                    SQL_EVAL_QAS_TABLE,SQL_EVAL_QAS_TABLE_SCHEMA,
                    SQL_DOC_TYPE_SUMMARIES_TABLE)
from utils.MySQLDB_manager import MySQLDB
from testing.modules.testing_modules import generate_questions

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

generate_questions(sql_con=sql_con,
                   table_eval_chunks=SQL_EVAL_CHUNKS_TABLE,
                   table_QAs=SQL_EVAL_QAS_TABLE,
                   table_QAs_schema=SQL_EVAL_QAS_TABLE_SCHEMA,
                   table_summaries = SQL_DOC_TYPE_SUMMARIES_TABLE,
                   overwrite=False)