from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

class PDFLoader:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def load_and_split_documents(self):
        pdf_data = []
        for path in self.pdf_paths:
            loader = PyPDFLoader(path)
            pdf_data.extend(loader.load())
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(pdf_data)
