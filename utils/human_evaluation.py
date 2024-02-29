import time
from src.chat_model import ChatModel
from utils.evaluation_metrics import ChatModelTester


class ChatModelTester:
    def __init__(self, chat_qa):
        self.chat_qa = chat_qa

    def human_evaluation(self, generated_answer, expected_answer):
        """
        Perform human evaluation of the generated answer.

        Parameters:
            generated_answer (str): The answer generated by the chat model.
            expected_answer (str): The expected answer from the test dataset.

        Returns:
            bool: True if the generated answer is acceptable according to human evaluation, False otherwise.
        """
        print("Generated Answer:", generated_answer)
        print("Expected Answer:", expected_answer)
        
        # Prompt human evaluator for feedback
        evaluation = input("Is the generated answer acceptable? (yes/no): ")
        
        return evaluation.lower() == "yes"

    def test_with_dataset(self, test_data):
        total_questions = len(test_data)
        correct_answers = 0
        
        for question, expected_answer in test_data:
            response = self.chat_qa({"question": question, "chat_history": []})
            generated_answer = response["answer"]
            
            # Perform human evaluation
            acceptable = self.human_evaluation(generated_answer, expected_answer)
            
            if acceptable:
                correct_answers += 1
                
        accuracy = (correct_answers / total_questions) * 100
        
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2f}%")

