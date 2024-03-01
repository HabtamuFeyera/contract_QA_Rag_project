import sys
import os
import getpass
from flask import Flask, render_template, request, jsonify
from src.pdf_loader import PDFLoader
from src.vector_embedding import VectorEmbedding
from src.chat_model import ChatModel

app = Flask(__name__, template_folder="/home/habte/contract_QA_Rag_project/frontend", static_folder="/home/habte/contract_QA_Rag_project/frontend")

# Define paths to PDF documents
pdf_paths = [
    "/home/habte/Downloads/Raptor Contract.docx.pdf",
    "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
    "/home/habte/Downloads/Robinson Advisory.docx.pdf",
    "/home/habte/Downloads/Robinson Q&A.docx.pdf"
]

# Prompt user to enter OpenAI API key
os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")

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

# Serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Handle the POST request for asking questions
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    response = chat_qa({"question": question})
    return jsonify({"response": response["answer"]})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)


