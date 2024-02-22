
import os
import nltk
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Download NLTK data
nltk.download('punkt')

def setup_embeddings(pdf_data):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get(''))
    # Use NLTK for tokenization
    split_data = [nltk.word_tokenize(doc) for doc in pdf_data]

    collection_name = "contracts_collection"
    persist_directory = "contracts_vect_embedding"
    vect_db = Chroma.from_documents(
        split_data,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    vect_db.persist()
    return vect_db
