import os
import getpass
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

class RAGEvaluation:
    def __init__(self, openai_api_key, short_contract_path, long_contract_path, questions_answers):
        self.chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.questions_answers = questions_answers
        self.short_contract_tokens = self._load_and_split_document(short_contract_path)
        self.long_contract_tokens = self._load_and_split_document(long_contract_path)
        self.short_contract_qa = ConversationalRetrievalChain.from_llm({
            "retriever": self.chat_model,
            "memory": self.memory,
            "tokens": self.short_contract_tokens
        })
        self.long_contract_qa = ConversationalRetrievalChain.from_llm({
            "retriever": self.chat_model,
            "memory": self.memory,
            "tokens": self.long_contract_tokens
        })

    def _load_and_split_document(self, contract_path):
        loader = PyPDFLoader(contract_path)
        contract_data = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(contract_data)

    def _evaluate_contract(self, contract_qa):
        total_questions = len(self.questions_answers)
        correct_answers = 0
        
        for question, correct_answer in self.questions_answers:
            response = contract_qa.query(question)
            generated_answer = response["text"]
            if generated_answer.strip().lower() == correct_answer.strip().lower():
                correct_answers += 1
        
        return correct_answers / total_questions

    def evaluate(self):
        short_contract_accuracy = self._evaluate_contract(self.short_contract_qa)
        long_contract_accuracy = self._evaluate_contract(self.long_contract_qa)
        return short_contract_accuracy, long_contract_accuracy

def main():
    # Prompt user to enter OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")

    # Set your OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Define paths to contract documents
    short_contract_path = "/home/habte/Downloads/Robinson Advisory.docx.pdf"
    long_contract_path = "/home/habte/Downloads/Raptor Contract.docx.pdf"

    # Define questions and correct answers
    questions_answers = [
        ("Question 1", "Correct answer 1"),
        ("Question 2", "Correct answer 2"),
        # Add remaining questions and correct answers
    ]

    # Initialize RAG evaluation
    rag_eval = RAGEvaluation(openai_api_key, short_contract_path, long_contract_path, questions_answers)

    # Perform evaluation
    short_contract_accuracy, long_contract_accuracy = rag_eval.evaluate()

    # Print evaluation results
    print("Short Contract Accuracy:", short_contract_accuracy)
    print("Long Contract Accuracy:", long_contract_accuracy)

if __name__ == "__main__":
    main()

