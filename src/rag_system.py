import os
from pdf_loader import PDFLoader
from vector_embedding import VectorEmbedding
from chat_model import ChatModel

class RAGSystem:
    def __init__(self, pdf_paths, openai_api_key):
        # Initialize PDF loader
        pdf_loader = PDFLoader(pdf_paths)
        split_data = pdf_loader.load_and_split_documents()

        # Initialize vector embedding
        vector_embedding = VectorEmbedding(openai_api_key)
        self.vect_db = vector_embedding.create_vector_store(split_data)

        # Initialize chat model
        chat_model = ChatModel(openai_api_key)
        self.chat_qa = chat_model.create_chat_qa(self.vect_db)

    def query(self, input_query):
        response = self.chat_qa({"question": input_query})
        return response["answer"]

if __name__ == "__main__":
    # Define paths to PDF documents
    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]

   
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Initialize RAG system
    rag_system = RAGSystem(pdf_paths, openai_api_key)

    # Example usage
    chat_history = []

    while True:
        query = input('You: ')  # Prompt the user to input a question

        if query.lower() == 'done':  # Check if the user wants to end the conversation
            break

        response = rag_system.query(query)  # Retrieve response

        chat_history.append({"role": "user", "content": query})  # Update chat history
        chat_history.append({"role": "assistant", "content": response})

        print('Assistant:', response)  # Print the assistant's response
