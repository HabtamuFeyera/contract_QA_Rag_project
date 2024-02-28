import os
import time
import getpass
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = getpass.getpass()

class RAGEvaluation:
    def __init__(self, openai_key):
        self.chat_model = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chat_qa = ConversationalRetrievalChain.from_llm(self.chat_model, memory=self.memory)
        self.evaluator_chain = RagasEvaluatorChain()
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

    def evaluate_with_ragas_metrics(self, queries, expected_responses):
        results = []
        for query, expected_response in zip(queries, expected_responses):
            generated_response = self.generate_response(query)
            result = self.evaluator_chain.evaluate(generated_response, expected_response)
            results.append(result)
        return results

def main():
    # Initialize RAG evaluation
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    rag_eval = RAGEvaluation(openai_key)
    
    # Example queries and expected responses
    queries = ["What is the contract about?", "Can you explain the terms?", "What is the payment schedule?"]
    expected_responses = ["The contract is about...", "The terms include...", "The payment schedule is..."]
    
    # Process queries and evaluate RAG system
    accuracy, avg_response_time = rag_eval.evaluate(queries, expected_responses)
    print(f"Accuracy: {accuracy}")
    print(f"Avg. Response Time: {avg_response_time} seconds")
    
    # Evaluate with Ragas metrics
    results = rag_eval.evaluate_with_ragas_metrics(queries, expected_responses)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
