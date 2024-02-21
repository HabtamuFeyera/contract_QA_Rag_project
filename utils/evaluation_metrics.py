from transformers import RagTokenizer, RagRetriever
import torch

# Load the RAG tokenizer and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("fine_tuned_retriever")

# Define evaluation data (questions and relevant documents)
evaluation_questions = ["What is the capital of France?", "Who wrote Hamlet?"]
relevant_documents = ["Paris is the capital of France.", "Hamlet is a play written by William Shakespeare."]

# Calculate Mean Reciprocal Rank (MRR)
def calculate_mrr(questions, relevant_documents, retriever, tokenizer):
    mrr = 0.0
    for question, relevant_document in zip(questions, relevant_documents):
        inputs = tokenizer(question, relevant_document, return_tensors="pt")
        outputs = retriever(**inputs)
        document_scores = outputs["document_scores"].detach().cpu().numpy()[0]
        ranked_indices = document_scores.argsort()[::-1]  # Sort document indices in descending order of scores
        rank_of_first_relevant_document = list(ranked_indices).index(0) + 1  # Rank of the first relevant document
        reciprocal_rank = 1 / rank_of_first_relevant_document if rank_of_first_relevant_document != 0 else 0
        mrr += reciprocal_rank
    mrr /= len(questions)  # Average MRR across all questions
    return mrr

# Calculate MRR
mrr = calculate_mrr(evaluation_questions, relevant_documents, retriever, tokenizer)
print("Mean Reciprocal Rank (MRR):", mrr)
