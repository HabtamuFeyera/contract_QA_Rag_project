from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

class ChatModel:
    def __init__(self, openai_api_key):
        """Initializes the ChatModel with the OpenAI API key."""
        self.chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)

    def create_chat_qa(self, vect_db):
        """
        Creates a conversational retrieval chain with memory.

        Args:
            vect_db: The vector database to use for retrieval.

        Returns:
            ConversationalRetrievalChain: The chat QA chain.
        """
        try:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            chat_qa = ConversationalRetrievalChain.from_llm(
                self.chat_model,
                vect_db.as_retriever(),  # Ensure vect_db provides the correct retriever
                memory=memory
            )
            return chat_qa
        except Exception as e:
            print(f"Error in creating conversational chain: {str(e)}")
            raise
