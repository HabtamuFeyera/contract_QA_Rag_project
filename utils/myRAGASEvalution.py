import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

class RAGEvaluation:
    def __init__(self, openai_key, pdf_paths):
        self.chat_model = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chat_qa = ConversationalRetrievalChain.from_llm(self.chat_model, self.vect_db.as_retriever(), memory=self.memory)
        self.total_queries = 0
        self.correct_responses = 0
        self.total_response_time = 0

    def evaluate(self, queries, expected_responses):
        for query, expected_response in zip(queries, expected_responses):
            start_time = time.time()
            generated_response = self.generate_response(query)
            end_time = time.time()

            self.total_queries += 1
            self.total_response_time += end_time - start_time

            is_correct = self.compare_responses(generated_response, expected_response)
            if is_correct:
                self.correct_responses += 1

        accuracy = self.correct_responses / self.total_queries
        avg_response_time = self.total_response_time / self.total_queries

        return accuracy, avg_response_time

    def generate_response(self, query):
        response = self.chat_qa.query(query)
        return response["text"]

    def compare_responses(self, response1, response2):
        return response1.strip().lower() == response2.strip().lower()

# Initialize RAG evaluation
openai_key = os.environ.get('OPENAI_API_KEY')
pdf_paths = [
    "/home/habte/Downloads/Raptor Contract.docx.pdf",
    "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
    "/home/habte/Downloads/Robinson Advisory.docx.pdf",
    "/home/habte/Downloads/Robinson Q&A.docx.pdf"
]

rag_eval = RAGEvaluation(openai_key, pdf_paths)

# Example queries and expected responses
queries = ["What is the contract about?", "Can you explain the terms?", "What is the payment schedule?"]
expected_responses = ["The contract is about...", "The terms include...", "The payment schedule is..."]

# Process queries and evaluate RAG system
accuracy, avg_response_time = rag_eval.evaluate(queries, expected_responses)
print(f"Accuracy: {accuracy}")
print(f"Avg. Response Time: {avg_response_time} seconds")