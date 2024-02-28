import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

class RAGEvaluation:
    def __init__(self, openai_key, short_contract_path, long_contract_path, questions_answers):
        self.chat_model = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.questions_answers = questions_answers
        
        # Load contracts
        short_contract_loader = PyPDFLoader(short_contract_path)
        long_contract_loader = PyPDFLoader(long_contract_path)
        short_contract_data = short_contract_loader.load()
        long_contract_data = long_contract_loader.load()
        
        # Split documents into tokens
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.short_contract_tokens = text_splitter.split_documents(short_contract_data)
        self.long_contract_tokens = text_splitter.split_documents(long_contract_data)
        
        # Initialize RAG systems
        self.short_contract_qa = ConversationalRetrievalChain.from_llm(self.chat_model, self.short_contract_tokens)
        self.long_contract_qa = ConversationalRetrievalChain.from_llm(self.chat_model, self.long_contract_tokens)

    def evaluate(self):
        total_questions = 0
        correct_answers = 0
        
        for question, correct_answer in self.questions_answers:
            total_questions += 1
            
            # Process question using short contract RAG system
            response = self.short_contract_qa.query(question)
            generated_answer = response["text"]
            
            # Compare generated answer with correct answer
            if generated_answer.strip().lower() == correct_answer.strip().lower():
                correct_answers += 1
        
        short_contract_accuracy = correct_answers / total_questions
        
        total_questions = 0
        correct_answers = 0
        
        for question, correct_answer in self.questions_answers:
            total_questions += 1
            
            # Process question using long contract RAG system
            response = self.long_contract_qa.query(question)
            generated_answer = response["text"]
            
            # Compare generated answer with correct answer
            if generated_answer.strip().lower() == correct_answer.strip().lower():
                correct_answers += 1
        
        long_contract_accuracy = correct_answers / total_questions
        
        return short_contract_accuracy, long_contract_accuracy

def main():
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
    openai_key = os.environ.get('OPENAI_API_KEY')
    rag_eval = RAGEvaluation(openai_key, short_contract_path, long_contract_path, questions_answers)
    
    # Perform evaluation
    short_contract_accuracy, long_contract_accuracy = rag_eval.evaluate()
    
    # Print evaluation results
    print("Short Contract Accuracy:", short_contract_accuracy)
    print("Long Contract Accuracy:", long_contract_accuracy)

if __name__ == "__main__":
    main()
