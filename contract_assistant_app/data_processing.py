
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

def load_documents(file_paths):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pdf_data = [loader.load() for loader in loaders]
    return pdf_data

def preprocess_text(text):
    # Add preprocessing steps as needed
    return text
