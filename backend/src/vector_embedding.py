import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class VectorEmbedding:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def create_vector_store(self, split_data):
        collection_name = "contracts_collection"
        local_directory = "contracts_vect_embedding"
        persist_directory = os.path.join(os.getcwd(), local_directory)
        vect_db = Chroma.from_documents(
            split_data,
            self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        vect_db.persist()
        return vect_db
