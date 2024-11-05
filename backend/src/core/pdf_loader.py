# src/core/pdf_loader.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

class PDFLoader:
    def __init__(self, pdf_paths):
        """
        Initializes the PDFLoader with a list of PDF file paths.
        
        Args:
            pdf_paths (list): A list of paths to PDF files to be loaded.
        """
        self.pdf_paths = pdf_paths

    def load_and_split_documents(self):
        """
        Loads and splits documents from the specified PDF files.
        
        Returns:
            list: A list of split document texts.
        """
        pdf_data = []
        
        # Load documents from each PDF path
        for path in self.pdf_paths:
            loader = PyPDFLoader(path)
            pdf_data.extend(loader.load())
        
        # Split the loaded documents into chunks
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(pdf_data)
