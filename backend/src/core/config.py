# src/core/config.py

import os

class Config:
    # Define configurations like API keys, database URLs, etc.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_default_key_here")
    VECTOR_STORE_PATH = "path/to/vector/store"
    PDF_DIRECTORY = "path/to/pdf/files"

config = Config()
