import os
from dotenv import load_dotenv
from db_operations import initialize_database

load_dotenv()

password = os.getenv("DB_PASSWORD")

if __name__ == "__main__":
    success = initialize_database(password=password)
    if success:
        print("Database initialized successfully!")
    else:
        print("Failed to initialize database.")