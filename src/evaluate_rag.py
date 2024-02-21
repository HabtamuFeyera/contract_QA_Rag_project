from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Load the fine-tuned RAG model
model = RagTokenForGeneration.from_pretrained("fine_tuned_rag_model")

# Define evaluation questions
evaluation_questions = [
    "What are the terms of payment in the contract?",
    "What is the termination clause in the contract?",
    # Add more evaluation questions as needed
]

# Generate responses using the RAG model
generated_responses = []
for question in evaluation_questions:
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, early_stopping=True)
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_responses.append(generated_response)

# Evaluate the generated responses
ground_truth_responses = [
    "The terms of payment include a 30-day payment period.",
    "The termination clause allows either party to terminate the contract with 30 days' notice.",
    # Add ground truth responses corresponding to evaluation questions
]

# Calculate evaluation metrics (e.g., BLEU score, ROUGE score)
# You can use libraries like NLTK or rouge_score for this purpose

# Example BLEU score calculation using NLTK
from nltk.translate.bleu_score import corpus_bleu
references = [[response.split()] for response in ground_truth_responses]
hypotheses = [response.split() for response in generated_responses]
bleu_score = corpus_bleu(references, hypotheses)

print("BLEU Score:", bleu_score)
