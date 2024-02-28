import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

class LangChainEvaluation:
    def __init__(self):
        self.total_queries = 0
        self.correct_responses = 0
        self.total_response_time = 0
        self.total_consistency_score = 0
        self.total_novelty_score = 0
        self.total_user_satisfaction_score = 0
        self.total_errors = 0

    def process_query(self, query, expected_response):
        start_time = time.time()
        generated_response = self.generate_response(query)
        end_time = time.time()

        self.total_queries += 1
        self.total_response_time += end_time - start_time

        is_correct = self.compare_responses(generated_response, expected_response)
        if is_correct:
            self.correct_responses += 1
        else:
            self.total_errors += 1

        # For demonstration purposes, scores are randomly generated
        self.total_consistency_score += 0.8  # Example consistency score
        self.total_novelty_score += 0.7  # Example novelty score
        self.total_user_satisfaction_score += 0.9  # Example user satisfaction score

    def generate_response(self, query):
        # Implement response generation logic here
        return "Generated response to query: " + query

    def compare_responses(self, response1, response2):
        # Implement response comparison logic here
        return response1 == response2

    def calculate_average_metrics(self):
        avg_accuracy = self.correct_responses / self.total_queries if self.total_queries > 0 else 0
        avg_response_time = self.total_response_time / self.total_queries if self.total_queries > 0 else 0
        avg_consistency_score = self.total_consistency_score / self.total_queries if self.total_queries > 0 else 0
        avg_novelty_score = self.total_novelty_score / self.total_queries if self.total_queries > 0 else 0
        avg_user_satisfaction_score = self.total_user_satisfaction_score / self.total_queries if self.total_queries > 0 else 0
        avg_error_rate = self.total_errors / self.total_queries if self.total_queries > 0 else 0

        return {
            "Accuracy": avg_accuracy,
            "Response Time": avg_response_time,
            "Consistency Score": avg_consistency_score,
            "Novelty Score": avg_novelty_score,
            "User Satisfaction Score": avg_user_satisfaction_score,
            "Error Rate": avg_error_rate
        }

# Initialize LangChain evaluation
langchain_eval = LangChainEvaluation()

# Load PDF documents and split into tokens
pdf_paths = [
    "/home/habte/Downloads/Raptor Contract.docx.pdf",
    "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
    "/home/habte/Downloads/Robinson Advisory.docx.pdf",
    "/home/habte/Downloads/Robinson Q&A.docx.pdf"
]

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

openai_key = os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vect_db = Chroma.from_documents(split_data, embeddings,
                                collection_name=collection_name,
                                persist_directory=persist_directory)
vect_db.persist()

# Initialize the OpenAI language model for conversation
chat_model = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_qa = ConversationalRetrievalChain.from_llm(chat_model, vect_db.as_retriever(), memory=memory)

# Example queries and expected responses
queries = ["What is the contract about?", "Can you explain the terms?", "What is the payment schedule?"]
expected_responses = ["The contract is about...", "The terms include...", "The payment schedule is..."]

# Process queries and evaluate LangChain system
for query, expected_response in zip(queries, expected_responses):
    langchain_eval.process_query(query, expected_response)

# Calculate and print average metrics
average_metrics = langchain_eval.calculate_average_metrics()
print("Average Metrics:")
for metric, value in average_metrics.items():
    print(f"{metric}: {value}")
