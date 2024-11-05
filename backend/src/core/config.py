# src/core/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to hold API keys and paths for the application.
    """


    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY must be set in the environment variables.")

    
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "path/to/vector/store")
    
    
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "path/to/pdf/files")


config = Config()
