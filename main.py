import os
os.chdir('/home/habte/contract_QA_Rag_project')  
from src.pdf_loader import PDFLoader
from src.vector_embedding import VectorEmbedding
from src.chat_model import ChatModel


# Set OpenAI API key
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
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

    # Initialize PDF loader
    pdf_loader = PDFLoader(pdf_paths)
    split_data = pdf_loader.load_and_split_documents()

    # Initialize vector embedding
    vector_embedding = VectorEmbedding(openai_api_key)
    vect_db = vector_embedding.create_vector_store(split_data)

    # Initialize chat model
    chat_model = ChatModel(openai_api_key)
    chat_qa = chat_model.create_chat_qa(vect_db)

    # Example usage
    chat_history = []

    while True:
        query = input('You: ')  # Prompt the user to input a question

        if query.lower() == 'done':  # Check if the user wants to end the conversation
            break

        response = chat_qa({"question": query, "chat_history": chat_history})  # Retrieve response

        chat_history.append({"role": "user", "content": query})  # Update chat history
        chat_history.append({"role": "assistant", "content": response["answer"]})

        print('Assistant:', response["answer"])  # Print the assistant's response
