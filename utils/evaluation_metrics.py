import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ChatModelTester:
    def __init__(self, chat_qa):
        self.chat_qa = chat_qa

    def test_with_dataset(self, test_data):
        total_questions = len(test_data)
        correct_answers = 0
        total_response_time = 0
        
        for question, expected_answer in test_data:
            start_time = time.time()
            response = self.chat_qa({"question": question, "chat_history": []})
            end_time = time.time()
            response_time = end_time - start_time
            total_response_time += response_time
            
            generated_answer = response["answer"]
            
            # Calculate BLEU score with smoothing function and specified weights
            smoothing_function = SmoothingFunction().method4
            bleu_score = sentence_bleu([expected_answer.split()], generated_answer.split(), smoothing_function=smoothing_function)
            
            # Define a threshold for BLEU score
            threshold = 0.5
            
            if bleu_score >= threshold:
                correct_answers += 1
                
            print(f"Question: {question}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"BLEU Score: {bleu_score}")
            print(f"Response Time: {response_time} seconds\n")
        
        accuracy = (correct_answers / total_questions) * 100
        average_response_time = total_response_time / total_questions
        
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Response Time: {average_response_time:.2f} seconds")
