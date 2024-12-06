import mysql.connector

class MySQLDB:
    def __init__(self, config, database_name):
        self.config = config
        self.database_name = database_name
        self.conn = mysql.connector.connect(**self.config)
        self.cursor = self.conn.cursor()
    
    def create_database(self):
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database_name}")

    def check_table_exists(self, table_name):
        """
        Checks if a table exists in the current database.

        :param table_name: Name of the table to check.
        :return: True if the table exists, False otherwise.
        """
        self.cursor.execute(f"USE {self.database_name}")
        query = """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = %s
        """
        self.cursor.execute(query, (self.database_name, table_name))
        result = self.cursor.fetchone()
        return result[0] > 0

    def create_table(self, table_name, schema):
        """
        Creates a table with the given schema, adds missing columns, and warns about extraneous columns.

        :param table_name: Name of the table.
        :param schema: A dictionary where keys are column names and values are data types.
        """
        # Step 1: Ensure database exists
        self.create_database()
        self.cursor.execute(f"USE {self.database_name}")

        # Step 2: Create table if it doesn't exist
        columns = ', '.join([f"{col_name} {col_type}" for col_name, col_type in schema.items()])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
        self.cursor.execute(create_table_sql)

        # Step 3: Check existing columns in the table
        self.cursor.execute(f"DESCRIBE {table_name}")
        existing_columns = {row[0]: row[1] for row in self.cursor.fetchall()}  # {column_name: data_type}

        # Step 4: Identify and handle mismatched data types
        for col_name, col_type in schema.items():
            simplified_type = col_type.split()[0].lower()  # Extract base type from expected schema
            # Treat BOOLEAN and TINYINT(1) as equivalent
            if simplified_type == "boolean" and existing_columns.get(col_name) == "tinyint(1)":
                continue  # Skip the warning for BOOLEAN vs TINYINT(1)
            if col_name not in existing_columns:
                # Add missing columns
                alter_table_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type};"
                self.cursor.execute(alter_table_sql)
                print(f"Added missing column: {col_name} ({col_type})")
            elif simplified_type != existing_columns[col_name]:
                # Warn about mismatched types
                print(f"Warning: Column '{col_name}' exists with a different type '{existing_columns[col_name]}' (expected: '{simplified_type}').")       
        
        # Step 5: Warn about extraneous columns
        for existing_col_name in existing_columns.keys():
            if existing_col_name not in schema:
                print(f"Warning: Column '{existing_col_name}' is present in the table but not in the provided schema. Consider removing it if obsolete.")

        self.conn.commit()
    
    def insert_record(self, table_name, record, overwrite=False):
        """
        Inserts a single record into the specified table. Optionally overwrites if a record with the same key exists.

        :param table_name: Name of the table.
        :param record: A dictionary where keys are column names and values are the data to insert.
        :param overwrite: If True, overwrites existing records with the same key. Defaults to False.
        """
        columns = ', '.join(record.keys())
        placeholders = ', '.join(['%s'] * len(record))
        values = tuple(record.values())

        if overwrite:
            # Use ON DUPLICATE KEY UPDATE to overwrite existing records
            update_clause = ', '.join([f"{col} = VALUES({col})" for col in record.keys()])
            query = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_clause}
            """
        else:
            # Regular INSERT, will raise an error if a duplicate key exists
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(query, values)
        self.conn.commit()
        return self.cursor.lastrowid

    def update_record(self, table_name, update_data, conditions, append=False):
        """
        Update a record in the specified table with multiple conditions. Optionally append new data to columns instead of replacing.

        :param table_name: Name of the table.
        :param update_data: A dictionary where keys are column names and values are the new data to update.
        :param conditions: A dictionary where keys are column names and values are the conditions to filter (e.g., {'chat_id': 123, 'message_id': 456}).
        :param append: If True, append new data to the columns instead of replacing. Default is False (replace).
        """

        # Build the SET clause for updating columns
        set_clause = []
        set_values = []

        for col, val in update_data.items():
            if append:
                # Append the value to the existing one if it's not null
                set_clause.append(f"{col} = CONCAT(IFNULL(CONCAT({col}, ', '), ''), %s)")
            else:
                # Normal update (replace) for other columns
                set_clause.append(f"{col} = %s")

            set_values.append(val)

        set_clause_str = ', '.join(set_clause)

        # Build the WHERE clause for the conditions
        where_clause = ' AND '.join([f"{col} = %s" for col in conditions.keys()])
        condition_values = list(conditions.values())

        # Combine SET values and condition values into one list for the query
        values = set_values + condition_values

        # Construct the query
        query = f"UPDATE {table_name} SET {set_clause_str} WHERE {where_clause}"

        # Execute the query
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(query, values)
        self.conn.commit()
    
    def insert_many_records(self, table_name, records, overwrite=False):
        """
        Inserts multiple records into the specified table with an option to overwrite existing rows.

        :param table_name: Name of the table.
        :param records: List of dictionaries, where each dictionary represents a single record.
        :param overwrite: If True, overwrite existing rows with new data. Defaults to False (ignore duplicates).
        :return: List of inserted record IDs.
        """
        if not records:
            return []
                    
        # Prepare column names and placeholders
        columns = ', '.join(records[0].keys())
        placeholders = ', '.join(['%s'] * len(records[0]))
        values = [tuple(record.values()) for record in records]

        # Use the current database
        self.cursor.execute(f"USE {self.database_name}")
        
        # Choose SQL behavior based on overwrite flag
        if overwrite:
            update_clause = ', '.join([f"{col} = VALUES({col})" for col in records[0].keys()])
            sql = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_clause}
            """
        else:
            sql = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        # Execute the query
        self.cursor.executemany(sql, values)
        self.conn.commit()

        # Get the current highest ID after insertion
        self.cursor.execute(f"SELECT LAST_INSERT_ID()")
        first_insert_id = self.cursor.fetchone()[0]

        # Calculate the range of inserted IDs
        return list(range(first_insert_id, first_insert_id + len(records)))
    
    def get_record(self, table_name, values, conditions):
        """
        Retrieve specific values from a table based on one or more conditions.

        :param table_name: Name of the table.
        :param values: A list of columns to retrieve (e.g., ['name', 'email']).
        :param conditions: A dictionary where keys are column names and values are the conditions (e.g., {'id': 123, 'status': 'active'}).
        :return: The first matching record as a tuple, or None if no record is found.
        """
        # Build the SELECT clause for the columns you want to retrieve
        select_clause = ', '.join(values)

        # Build the WHERE clause for the conditions
        where_clause = ' AND '.join([f"{col} = %s" for col in conditions.keys()])
        condition_values = list(conditions.values())

        # Construct the query
        query = f"SELECT {select_clause} FROM {table_name} WHERE {where_clause}"

        # Execute the query
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(query, condition_values)

        # Fetch the first matching record
        result = self.cursor.fetchone()

        # Return the value (not a tuple)
        if result:
            return result  # This will return just the value instead of a tuple

        return None
    
    def get_records(self, table_name, values, conditions):
        """
        Retrieve all records matching specific values and conditions, returning as a list of dictionaries.
        """
        # Build the SELECT clause
        select_clause = ', '.join(values)

        # Prepare the WHERE clause
        where_clauses = []
        condition_values = []

        for col, val in conditions.items():
            if isinstance(val, (list, tuple)):  # Handle IN clause
                placeholders = ', '.join(['%s'] * len(val))
                where_clauses.append(f"{col} IN ({placeholders})")
                condition_values.extend(val)
            else:  # Handle standard equality
                where_clauses.append(f"{col} = %s")
                condition_values.append(val)

        where_clause = ' AND '.join(where_clauses)

        # Construct the query
        query = f"SELECT {select_clause} FROM {table_name} WHERE {where_clause}"

        # Execute the query
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(query, condition_values)

        # Fetch all records
        rows = self.cursor.fetchall()

        # Convert to list of dictionaries
        return [dict(zip(values, row)) for row in rows]
    
    def get_record_by_id(self, table_name, record_id):
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE id = %s", (record_id,))
        return self.cursor.fetchone()
    
    def get_records_by_ids(self, table_name, record_ids):
        self.cursor.execute(f"USE {self.database_name}")
        format_strings = ','.join(['%s'] * len(record_ids))
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE id IN ({format_strings})", tuple(record_ids))
        # Get the column names from the cursor
        rows = self.cursor.fetchall()
        columns = [column[0] for column in self.cursor.description]
        # Convert rows to list of dictionaries
        records = [dict(zip(columns, row)) for row in rows]
        return records
    
    def delete_table(self, table_name):
        """
        Deletes the specified table from the database.

        :param table_name: Name of the table to delete.
        """
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.commit()

    def find_record_id(self, table_name, criteria):
        """
        Finds the ID of a record in the specified table based on given criteria.

        :param table_name: Name of the table.
        :param criteria: A dictionary where keys are column names and values are the criteria to search for.
        :return: The ID of the record if found, None otherwise.
        """
        columns = ' AND '.join([f"{col} = %s" for col in criteria.keys()])
        values = tuple(criteria.values())
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(f"SELECT id FROM {table_name} WHERE {columns}", values)
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def get_all_records(self, table_name):
        """
        Retrieves all records from the specified table.

        :param table_name: Name of the table to retrieve records from.
        :return: A list of tuples, where each tuple represents a row in the table.
        """
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(f"SELECT * FROM {table_name} order by id")
        return self.cursor.fetchall()
    
    def get_all_records_as_dict(self, table_name):
        """
        Retrieves all records from the specified table and returns them as a list of dictionaries.

        :param table_name: Name of the table to retrieve records from.
        :return: A list of dictionaries, where each dictionary represents a row in the table.
        """
        self.cursor.execute(f"USE {self.database_name}")
        self.cursor.execute(f"SELECT * FROM {table_name}")
        
        # Get the column names from the cursor
        columns = [column[0] for column in self.cursor.description]
        
        # Fetch all rows from the table
        rows = self.cursor.fetchall()
        
        # Convert rows to list of dictionaries
        records = [dict(zip(columns, row)) for row in rows]
        
        return records
    
    def check_table_and_count(self, table_name):
        """
        Checks if a table exists and returns the number of records in it.

        :param table_name: Name of the table to check.
        :return: A tuple (exists, count), where `exists` is True if the table exists and `count` is the number of records.
                If the table does not exist, `count` will be None.
        """
        try:
            # Check if the table exists
            self.cursor.execute(f"USE {self.database_name}")
            query = """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_name = %s
            """
            self.cursor.execute(query, (self.database_name, table_name))
            exists = self.cursor.fetchone()[0] > 0

            if not exists:
                return False, None

            # Count the number of records if the table exists
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cursor.fetchone()[0]
            return True, count
        except Exception as e:
            print(f"Error checking table or counting records: {e}")
            return False, None
    
    def close(self):
        self.cursor.close()
        self.conn.close()

