from telethon import TelegramClient

# Your Telegram API credentials
API_ID = '21404926'
API_HASH = 'a1d4678be38504d7584bd5b600f91005'
BOT_USERNAME = '@rodolfocco'  # Replace with the bot's username, e.g., "@your_bot"

# Create a Telethon client
client = TelegramClient('session_name', API_ID, API_HASH,)

async def send_messages():
    # Start the client
    await client.start()
    
    # List of questions to send
    questions = [
        "Servus",
        "Wie heißt du?",
        "What are the funding requirements?",
    ]

    # Send each question to the bot
    for question in questions:
        await client.send_message(BOT_USERNAME, question)
        print(f"Sent: {question}")

# Run the client
client.loop.run_until_complete(send_messages())