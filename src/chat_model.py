from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class ChatModel:
    def __init__(self, openai_api_key):
        #self.chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        self.chat_model= ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.0)

    def create_chat_qa(self, vect_db):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_qa = ConversationalRetrievalChain.from_llm(
            self.chat_model,
            vect_db.as_retriever(),
            memory=memory
        )
        return chat_qa
