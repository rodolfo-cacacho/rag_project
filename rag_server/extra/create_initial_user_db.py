import mysql.connector
import csv
import os

# MySQL connection details
CONFIG_SQL_DB = {
    'user': 'root',            # Replace with your MySQL username
    'password': 'admin123',     # Replace with your MySQL password
    'host': 'localhost'         # Adjust if necessary
}

DB_NAME = 'data_rag'  # Replace with your actual database name
CSV_FILE_PATH = 'conversations/users_db.csv'  # Replace with your CSV file path

USER_DB_TABLE = 'user_db'


# Users table schema
users_table_schema = {
    'id': 'BIGINT PRIMARY KEY',
    'username': 'VARCHAR(100) NULL',
    'name': 'VARCHAR(100)',
    'date_added': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'date_edit': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP',
    'allowed': 'BOOLEAN DEFAULT FALSE'
}

# Create a MySQL connection
def connect_db(config, db_name):
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    cursor.execute(f"USE {db_name}")
    return conn, cursor

# Create the Users table using the provided schema
def create_users_table(cursor, schema,table_name=USER_DB_TABLE):
    columns = ', '.join([f"{col_name} {col_type}" for col_name, col_type in schema.items()])
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
    cursor.execute(query)

# Insert data into the Users table
def insert_user_data(cursor, user_data,table_name=USER_DB_TABLE):
    insert_query = f"""
    INSERT INTO {table_name} (date_added, username, id, name, date_edit, allowed)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        username = VALUES(username),
        name = VALUES(name),
        date_edit = VALUES(date_edit),
        allowed = VALUES(allowed)
    """
    cursor.execute(insert_query, (
        user_data['date_added'],
        user_data['username'],
        user_data['id'],
        user_data['name'],
        user_data['date_edit'],
        user_data['allowed']
    ))

# Read the CSV file and insert the data into the database
def read_csv_and_insert_data(cursor, csv_file_path):
    if not os.path.isfile(csv_file_path):
        print(f"CSV file {csv_file_path} does not exist.")
        return
    
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)  # This reads the CSV as a dictionary
        for row in reader:
            # Ensure the necessary fields are present, and fill missing ones if needed
            user_data = {
                'date_added': row.get('date_added', None),
                'username': row.get('username', None),
                'id': row.get('id', None),
                'name': row.get('name', None),
                'date_edit': row.get('date_edit', None),
                # Convert 'allowed' field from string to boolean
                'allowed': row.get('allowed', 'False').strip().lower() == 'true'
            }

            # Insert each user into the database
            if user_data['id']:  # Ensure there is an ID present
                insert_user_data(cursor, user_data)

# Main execution
if __name__ == "__main__":
    # Connect to the database
    conn, cursor = connect_db(CONFIG_SQL_DB, DB_NAME)
    
    # Create the Users table using the provided schema
    create_users_table(cursor, users_table_schema)
    
    # Read the CSV file and insert data
    read_csv_and_insert_data(cursor, CSV_FILE_PATH)
    
    # Commit and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Users table created and data inserted successfully!")