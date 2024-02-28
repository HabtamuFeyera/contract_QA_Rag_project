import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

class RAGSystem:
    def __init__(self, pdf_paths, openai_api_key):
        # Load PDF documents and split into tokens
        pdf_data = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pdf_data.extend(loader.load())

        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_data = text_splitter.split_documents(pdf_data)

        # Set up ChromaDB with vector embeddings
        collection_name = "contracts_collection"
        local_directory = "contracts_vect_embedding"
        persist_directory = os.path.join(os.getcwd(), local_directory)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vect_db = Chroma.from_documents(
            split_data,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.vect_db.persist()

        # Initialize ConversationalRetrievalChain
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chat_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo"),
            self.vect_db.as_retriever(),
            memory=self.memory
        )

    def query(self, input_query):
        # Process user query and generate response
        response = self.chat_qa.query(input_query)
        return response["text"]

# Example usage
if __name__ == "__main__":
    # Define paths to PDF documents
    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]

    # Set your OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Initialize RAG system
    rag_system = RAGSystem(pdf_paths, openai_api_key)

    # User query
    user_query = "What is the contract about?"

    # Get response from RAG system
    response = rag_system.query(user_query)
    print("Response:", response)
