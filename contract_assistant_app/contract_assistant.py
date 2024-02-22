
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

def setup_contract_assistant():
    # Paths to the PDF files
    pdf_paths = [
        "/home/habte/Downloads/Raptor Contract.docx.pdf",
        "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
        "/home/habte/Downloads/Robinson Advisory.docx.pdf",
        "/home/habte/Downloads/Robinson Q&A.docx.pdf"
    ]
    
    loaders = [PyPDFLoader(path) for path in pdf_paths]
    pdf_data = [loader.load() for loader in loaders]

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Flatten the list of lists
    pdf_data_flat = [page for doc in pdf_data for page in doc]
    split_data = text_splitter.split_documents(pdf_data_flat)  # Use split_documents here

    collection_name = "contracts_collection"
    local_directory = "contracts_vect_embedding"
    persist_directory = os.path.join(os.getcwd(), local_directory)

    openai_api_key = ''  
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vect_db = Chroma.from_documents(
        split_data,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    vect_db.persist()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo"),
        vect_db.as_retriever(),
        memory=memory
    )
    return chat_qa

def get_response_to_query(query):
    assistant = setup_contract_assistant()
    response = assistant.generate_response(query) 
    return response
