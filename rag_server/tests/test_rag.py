import csv
import time
import telebot
from statistics import mean

# Initialize the bot with your API key
bot = telebot.TeleBot('YOUR_API_KEY_HERE')

# User ID of the recipient for the summary
SUMMARY_USER_ID = 'RECIPIENT_USER_ID_HERE'

# Global variable to capture the time when a question was sent
question_start_time = None
response_times = []

# Function to read questions from a CSV file
def read_questions_from_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        questions = [row[0] for row in reader]
    return questions

# Handler to capture the bot's response and calculate response time
@bot.message_handler(func=lambda message: True)
def handle_response(message):
    global question_start_time
    if question_start_time:
        response_time = time.time() - question_start_time
        response_times.append(response_time)
        question_start_time = None  # Reset for the next question
        print(f"Response received in {response_time:.2f} seconds: {message.text}")

# Function to send a question and wait for the response
def ask_question_and_wait(question):
    global question_start_time
    question_start_time = time.time()
    bot.send_message(SUMMARY_USER_ID, question)
    print(f"Question sent: {question}")

# Main function to run the test
def run_rag_test(file_path, n_questions):
    questions = read_questions_from_csv(file_path)

    # If there are fewer questions than requested, adjust n_questions
    n_questions = min(n_questions, len(questions))

    # Select n_questions from the list
    selected_questions = questions[:n_questions]

    # Send the /start command to initiate conversation
    bot.send_message(SUMMARY_USER_ID, "/start")
    time.sleep(1)  # Give the bot some time to respond to /start

    # Ask each question
    for question in selected_questions:
        ask_question_and_wait(question)
        time.sleep(2)  # Adjust sleep to allow time for response handling

    # Calculate summary statistics
    avg_time = mean(response_times)
    max_time = max(response_times)
    min_time = min(response_times)
    total_questions = len(selected_questions)

    # Send summary to the user
    summary_message = (
        f"RAG System Test Summary:\n"
        f"Total Questions Asked: {total_questions}\n"
        f"Average Time per Question: {avg_time:.2f} seconds\n"
        f"Max Time for a Question: {max_time:.2f} seconds\n"
        f"Min Time for a Question: {min_time:.2f} seconds\n"
    )
    bot.send_message(SUMMARY_USER_ID, summary_message)

if __name__ == "__main__":
    # Specify the CSV file path and the number of questions to ask
    csv_file_path = 'questions.csv'
    number_of_questions = 5  # Change this number as needed

    # Run the test
    run_rag_test(csv_file_path, number_of_questions)

    # Keep polling for responses
    bot.polling()