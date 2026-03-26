import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")