"""
PDF and Contract Document Ingestion Loader.
Extracts structured text from legal contracts preserving page numbers and file source metadata.
"""

import logging
import os
from pathlib import Path
from typing import List, Union, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads and preprocesses PDF contracts with metadata extraction."""

    def __init__(self, pdf_paths: Optional[Union[str, List[str]]] = None):
        if pdf_paths is None:
            self.pdf_paths = []
        elif isinstance(pdf_paths, str):
            self.pdf_paths = [pdf_paths]
        else:
            self.pdf_paths = pdf_paths

    def load_documents(self) -> List[Document]:
        """
        Loads all PDF documents from specified file paths.
        
        Returns:
            List[Document]: List of LangChain Documents with page-level text and metadata.
        """
        all_docs: List[Document] = []

        for path_str in self.pdf_paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"Contract file not found at path: {path}")
                continue

            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(path))
                pages = loader.load()

                filename = path.name
                for page in pages:
                    # Clean and enrich metadata
                    page_num = page.metadata.get("page", 0) + 1
                    page.metadata["source"] = filename
                    page.metadata["page_number"] = page_num
                    page.metadata["file_path"] = str(path)

                all_docs.extend(pages)
                logger.info(f"Loaded {len(pages)} pages from {filename}")
            except Exception as e:
                logger.error(f"Failed to load PDF {path}: {str(e)}")

        return all_docs

    @staticmethod
    def load_from_directory(directory_path: str) -> List[Document]:
        """Loads all PDF files found within a specified directory."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory_path}")
            return []

        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))
        loader = PDFLoader([str(p) for p in pdf_files])
        return loader.load_documents()

    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
        """Extracts document pages directly from raw uploaded bytes."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader = PDFLoader(tmp_path)
            docs = loader.load_documents()
            for doc in docs:
                doc.metadata["source"] = filename
            return docs
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
