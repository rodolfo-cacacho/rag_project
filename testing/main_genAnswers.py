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
                    SQL_EVAL_QAS_TABLE,
                    SQL_EVAL_QAS_TABLE_SCHEMA)
from utils.MySQLDB_manager import MySQLDB
from testing.modules.testing_modules import generate_answers,generate_true_answers

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

print("HERE")
sql_con.create_table(SQL_EVAL_QAS_TABLE,SQL_EVAL_QAS_TABLE_SCHEMA)

generate_answers(sql_con=sql_con,
                   table_eval_chunks=SQL_EVAL_CHUNKS_TABLE,
                   table_QAs=SQL_EVAL_QAS_TABLE)

generate_true_answers(sql_con=sql_con,
                   table_eval_chunks=SQL_EVAL_CHUNKS_TABLE,
                   table_QAs=SQL_EVAL_QAS_TABLE)