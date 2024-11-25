import csv
import os
from datetime import datetime

# Function to add a user to the CSV file
def add_user_to_csv(file_path, username, id, name, allowed=True):
    headers = ['date_added', 'username', 'id', 'name', 'date_edit', 'allowed']
    file_exists = os.path.isfile(file_path)
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    updated = False
    id = str(id)
    if file_exists:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            for row in rows:
                print(f'check: {row['id']==id}')
                if row['id'] == id:
                    row['username'] = username
                    row['name'] = name
                    row['date_edit'] = current_date
                    row['allowed'] = allowed
                    updated = True
            if updated:
                print(f'printing rows:\n{rows}')
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
    if not updated:
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'date_added': current_date,
                'username': username,
                'id': id,
                'name': name,
                'date_edit': current_date,
                'allowed': allowed
            })

# Function to check if a user is allowed
def is_user_allowed(file_path,telegram_username='None', telegram_id=''):
    if not os.path.isfile(file_path):
        return 'False', 'False'

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['username'] == telegram_username or (row['username'] == '' and row['id'] == str(telegram_id)):
                return 'True', row['allowed']
    return 'False', 'False'

def get_allowed_users(file_path):
    allowed_users = []
    if not os.path.isfile(file_path):
        return allowed_users

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['allowed'] == 'True':
                user_info = row['username'] if row['username'] != '' else row['id']
                allowed_users.append(user_info)
    return allowed_users