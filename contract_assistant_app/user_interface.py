
from data_processing import load_documents, preprocess_text
from nlp_models import setup_embeddings
from contract_assistant import get_response_to_query

def main():
    file_paths = ["/home/habte/Downloads/Raptor Contract.docx.pdf",
                  "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
                  "/home/habte/Downloads/Robinson Advisory.docx.pdf",
                  "/home/habte/Downloads/Robinson Q&A.docx.pdf"]
    
    # Load documents
    pdf_data = load_documents(file_paths)

    # Preprocess text
    preprocessed_data = [preprocess_text(doc) for doc in pdf_data]

    # Setup embeddings
    vect_db = setup_embeddings(preprocessed_data)


    query = input("Enter your query: ")
    response = get_response_to_query(query)
    print("Assistant response:", response)

if __name__ == "__main__":
    main()
