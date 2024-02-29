import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

class ConversationEvaluator:
    def __init__(self, pdf_paths, openai_api_key):
        self.vector_store = self._create_vector_store(pdf_paths, openai_api_key)
        self.chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_chain = self._create_conversation_chain()

    def _create_vector_store(self, pdf_paths, openai_api_key):
        pdf_data = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pdf_data.extend(loader.load())

        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_data = text_splitter.split_documents(pdf_data)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        collection_name = "contracts_collection"
        persist_directory = os.path.join(os.getcwd(), "contracts_vect_embedding")
        vect_db = Chroma.from_documents(split_data, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        vect_db.persist()

        return vect_db

    def _create_conversation_chain(self):
        return ConversationalRetrievalChain.from_llm(self.chat_model, self.vector_store.as_retriever(), memory=self.memory)

    def evaluate_relevance(self, queries, ground_truth):
        total_queries = len(queries)
        total_relevance = 0

        for query, expected_answer in zip(queries, ground_truth):
            response = self.conversation_chain({"question": query})["answer"]
            relevance = self._calculate_relevance(response, expected_answer)
            total_relevance += relevance

        avg_relevance = total_relevance / total_queries
        return avg_relevance

    def _calculate_relevance(self, response, expected_answer):
        response_embedding = self.vector_store({"text": response})["vector"]
        expected_answer_embedding = self.vector_store({"text": expected_answer})["vector"]
        similarity = cosine_similarity([response_embedding], [expected_answer_embedding])[0][0]
        return similarity

    def evaluate_response_time(self, queries):
        total_response_time = 0

        for query in queries:
            start_time = time.time()
            response = self.conversation_chain({"question": query})["answer"]
            end_time = time.time()
            total_response_time += end_time - start_time

        avg_response_time = total_response_time / len(queries)
        return avg_response_time

# Example usage
if __name__ == "__main__":
    # Prompt user to enter OpenAI API key
    openai_api_key = input("Enter your OpenAI API key: ")

    # Define paths to PDF documents
    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]

    # Create ConversationEvaluator instance
    evaluator = ConversationEvaluator(pdf_paths, openai_api_key)

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

    # Evaluate relevance
    avg_relevance = evaluator.evaluate_relevance(queries, ground_truth)
    print("Average Relevance:", avg_relevance)

    # Evaluate response time
    avg_response_time = evaluator.evaluate_response_time(queries)
    print("Average Response Time:", avg_response_time)
