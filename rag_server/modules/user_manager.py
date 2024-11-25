from config import SQL_USER_TABLE
from datetime import datetime


# Function to add or update a user in the database
def add_user_to_db(db, username, user_id, name, allowed=True,user_db_table = SQL_USER_TABLE):
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the user already exists in the database
    existing_user = db.get_record_by_id(user_db_table, user_id)
    
    if existing_user:
        # Update the user if they exist
        update_data = {
            'username': username,
            'name': name,
            'date_edit': current_date,
            'allowed': allowed
        }
        db.update_record(user_db_table, user_id, update_data)
    else:
        # Add a new user if they don't exist
        record = {
            'id': user_id,
            'username': username,
            'name': name,
            'date_added': current_date,
            'date_edit': current_date,
            'allowed': allowed
        }
        db.insert_record(user_db_table, record)

# Function to check if a user is allowed (MySQL-based)
def is_user_allowed(db, telegram_username='None', telegram_id='', user_db_table=SQL_USER_TABLE):
    query = f"""
        SELECT allowed FROM {user_db_table}
        WHERE username = %s OR (username = '' AND id = %s)
    """
    db.cursor.execute(f"USE {db.database_name}")
    db.cursor.execute(query, (telegram_username, telegram_id))
    result = db.cursor.fetchone()
    
    if result:
        # Convert 1/0 from the database to 'True'/'False' strings
        return 'True', 'True' if result[0] == 1 else 'False'
    
    return 'False', 'False'

# Function to get all allowed users
def get_allowed_users(db,user_db_table = SQL_USER_TABLE):
    allowed_users = []
    query = f"SELECT username, id FROM {user_db_table} WHERE allowed = TRUE"
    
    db.cursor.execute(f"USE {db.database_name}")
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    
    for row in rows:
        username = row[0] if row[0] != '' else str(row[1])
        allowed_users.append(username)
        
    return allowed_users