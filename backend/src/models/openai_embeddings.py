# openai_embeddings.py
from langchain.embeddings.openai import OpenAIEmbeddings

class OpenAIEmbeddingsWrapper:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def encode(self, document):
        return self.embeddings.encode(document)
