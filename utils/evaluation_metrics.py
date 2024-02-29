import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class RAGSystemEvaluator:
    def __init__(self, chat_qa):
        self.chat_qa = chat_qa

    def evaluate_with_dataset(self, test_data):
        total_questions = len(test_data)
        correct_answers = 0
        total_response_time = 0
        bleu_scores = []
        consistency_count = 0
        previous_answer = None

        for question, expected_answer in test_data:
            start_time = time.time()
            response = self.chat_qa({"question": question, "chat_history": []})
            end_time = time.time()
            response_time = end_time - start_time
            total_response_time += response_time

            generated_answer = response["answer"]

            # Calculate BLEU score
            smoothing_function = SmoothingFunction().method4
            bleu_score = sentence_bleu([expected_answer.split()], generated_answer.split(), smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)

            # Check accuracy
            if generated_answer == expected_answer:
                correct_answers += 1

            # Check consistency
            if previous_answer is not None and generated_answer == previous_answer:
                consistency_count += 1
            previous_answer = generated_answer

            print(f"Question: {question}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"BLEU Score: {bleu_score}")
            print(f"Response Time: {response_time} seconds\n")

        accuracy = (correct_answers / total_questions) * 100
        relevance = accuracy  # Relevance same as accuracy
        average_response_time = total_response_time / total_questions
        consistency = (consistency_count / total_questions) * 100
        average_bleu_score = sum(bleu_scores) / len(bleu_scores)

        evaluation_results = {
            "Total Questions": total_questions,
            "Correct Answers": correct_answers,
            "Accuracy": f"{accuracy:.2f}%",
            "Relevance": f"{relevance:.2f}%",
            "Average Response Time": f"{average_response_time:.2f} seconds",
            "Consistency": f"{consistency:.2f}%",
            "Average BLEU Score": average_bleu_score
        }

        return evaluation_results
