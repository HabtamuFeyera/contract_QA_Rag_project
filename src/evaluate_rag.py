import os
import getpass
from pdf_loader import PDFLoader
from vector_embedding import VectorEmbedding
from chat_model import ChatModel
from lang_model import LanguageModel  # Import your language model module

class LangChainEvaluation:
    def __init__(self, pdf_paths, openai_api_key):
        # Initialize PDF loader
        self.pdf_loader = PDFLoader(pdf_paths)
        self.split_data = self.pdf_loader.load_and_split_documents()

        # Initialize vector embedding
        self.vector_embedding = VectorEmbedding(openai_api_key)
        self.vect_db = self.vector_embedding.create_vector_store(self.split_data)

        # Initialize chat model
        self.chat_model = ChatModel(openai_api_key)
        self.chat_qa = self.chat_model.create_chat_qa(self.vect_db)

        # Initialize language model
        self.language_model = LanguageModel(openai_api_key)

    def generate_example(self):
        # Generate an example using your language model
        example = self.language_model.generate_example()
        return example

    def manual_evaluation(self, query):
        # Manual evaluation and debugging
        response = self.query(query)
        print("Manual Evaluation - Query:", query)
        print("Manual Evaluation - Response:", response)

    def llm_assisted_evaluation(self, query):
        # LLM-assisted evaluation
        response = self.query(query)
        llm_evaluation = self.language_model.evaluate_example(response)
        print("LLM-assisted Evaluation - Query:", query)
        print("LLM-assisted Evaluation - LLM Evaluation:", llm_evaluation)

    def query(self, input_query):
        response = self.chat_qa({"question": input_query})
        return response["answer"]

if __name__ == "__main__":
    # Prompt user to enter OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")

    # Set your OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Define paths to PDF documents
    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]

    # Initialize LangChain Evaluation platform
    langchain_eval = LangChainEvaluation(pdf_paths, openai_api_key)

    # Example generation
    print("Generating example...")
    example = langchain_eval.generate_example()
    print("Generated Example:", example)

    # Manual evaluation and debugging
    query = "What is the contract about?"
    langchain_eval.manual_evaluation(query)

    # LLM-assisted evaluation
    langchain_eval.llm_assisted_evaluation(query)
