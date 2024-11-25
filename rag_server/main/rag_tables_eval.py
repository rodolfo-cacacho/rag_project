import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.MySQLDB_manager import MySQLDB
from utils.json_table_import import import_table
from config import (CONFIG_SQL_DB,DB_NAME,SQL_VOCAB_BM25_TABLE,SQL_CHUNK_TABLE,
                    SQL_CHUNK_TABLE_SCHEMA,SQL_VOCAB_BM25_TABLE_SCHEMA,
                    TABLE_STORE_DIR)

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

success = 0
success_exp = 2

if not sql_con.check_table_exists(SQL_VOCAB_BM25_TABLE):

    table_name = SQL_VOCAB_BM25_TABLE+'.json'
    table_path = os.path.join(TABLE_STORE_DIR,table_name)
    if os.path.exists(table_path):
        loading_dict = import_table(table_path)

        ## Create and import table
        print("Created BM25 table")
        sql_con.create_table(SQL_VOCAB_BM25_TABLE,SQL_VOCAB_BM25_TABLE_SCHEMA)

        sql_con.insert_many_records(SQL_VOCAB_BM25_TABLE,loading_dict,overwrite=False)
        success += 1

    else:
        print("JSON file not found")
else:
    success+=1


if not sql_con.check_table_exists(SQL_CHUNK_TABLE):

    table_name = SQL_CHUNK_TABLE+'.json'
    table_path = os.path.join(TABLE_STORE_DIR,table_name)
    if os.path.exists(table_path):

        loading_dict = import_table(table_path)

        ## Create and import table
        print("Created Chunks table")
        sql_con.create_table(SQL_CHUNK_TABLE,SQL_CHUNK_TABLE_SCHEMA)

        sql_con.insert_many_records(SQL_CHUNK_TABLE,loading_dict,overwrite=False)
        success += 1
    
    else:
        print(f"JSON file {table_name} not found")

else:
    success+=1

print(f"DB Checkup completed {success}/{success_exp}")
