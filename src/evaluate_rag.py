import os
import getpass
import time
from pdf_loader import PDFLoader
from vector_embedding import VectorEmbedding
from chat_model import ChatModel
from sklearn.metrics.pairwise import cosine_similarity

class RAGEvaluation:
    def __init__(self, pdf_paths, openai_api_key):
        self.rag_system = self._initialize_rag_system(pdf_paths, openai_api_key)

    def _initialize_rag_system(self, pdf_paths, openai_api_key):
        # Initialize PDF loader
        pdf_loader = PDFLoader(pdf_paths)
        split_data = pdf_loader.load_and_split_documents()

        # Initialize vector embedding
        vector_embedding = VectorEmbedding(openai_api_key)
        vect_db = vector_embedding.create_vector_store(split_data)

        # Initialize chat model
        chat_model = ChatModel(openai_api_key)
        chat_qa = chat_model.create_chat_qa(vect_db)

        return chat_qa

    def evaluate(self, queries, ground_truth):
        total_queries = len(queries)
        total_relevance = 0
        total_response_time = 0

        for query, expected_answer in zip(queries, ground_truth):
            start_time = time.time()
            response = self.rag_system({"question": query})["answer"]
            end_time = time.time()
            total_response_time += end_time - start_time

            # Calculate relevance
            relevance = self.calculate_relevance(response, expected_answer)
            total_relevance += relevance

        # Calculate average relevance and response time
        avg_relevance = total_relevance / total_queries
        avg_response_time = total_response_time / total_queries

        return avg_relevance, avg_response_time

    def calculate_relevance(self, response, expected_answer):
        # Convert response and expected answer to embeddings
        response_embedding = self.rag_system({"question": response})["vector"]
        expected_answer_embedding = self.rag_system({"question": expected_answer})["vector"]

        # Calculate cosine similarity between embeddings
        similarity = cosine_similarity([response_embedding], [expected_answer_embedding])[0][0]

        return similarity

# Example usage
if __name__ == "__main__":
    # Prompt user to enter OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Define paths to PDF documents
    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]

    # Create RAGEvaluation instance
    rag_evaluation = RAGEvaluation(pdf_paths, openai_api_key)

    # Define queries and ground truth answers for evaluation
    queries = [
        "How much is the escrow amount?",
        "Is any of the Sellers bound by a non-competition covenant after the Closing?",
        "What is the termination notice?"
    ]
    ground_truth = [
        "The escrow amount is equal to $1,000,000",
        "No",
        "According to section 4:14 days..."
    ]

    # Evaluate RAG system
    avg_relevance, avg_response_time = rag_evaluation.evaluate(queries, ground_truth)

    print("Average Relevance:", avg_relevance)
    print("Average Response Time:", avg_response_time)
