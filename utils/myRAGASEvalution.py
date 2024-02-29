import os
import time
import getpass
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


class RAGEvaluation:
    def __init__(self, openai_key, pdf_paths):
        self.chat_model = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Initialize other necessary components (e.g., vector stores) if required
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

        accuracy = self.correct_responses / self.total_queries if self.total_queries > 0 else 0
        avg_response_time = self.total_response_time / self.total_queries if self.total_queries > 0 else 0

        return accuracy, avg_response_time

    def generate_response(self, query):
        response = self.chat_model.query(query)
        return response["text"]

    def compare_responses(self, response1, response2):
        return response1.strip().lower() == response2.strip().lower()

def main():
    # Set OpenAI API key
    openai_key = getpass.getpass("Enter your OpenAI API key: ")

    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]

    rag_eval = RAGEvaluation(openai_key, pdf_paths)

    queries = ["What is the contract about?", "Can you explain the terms?", "What is the payment schedule?"]
    expected_responses = ["The contract is about...", "The terms include...", "The payment schedule is..."]

    accuracy, avg_response_time = rag_eval.evaluate(queries, expected_responses)
    print(f"Accuracy: {accuracy}")
    print(f"Avg. Response Time: {avg_response_time} seconds")

if __name__ == "__main__":
    main()
