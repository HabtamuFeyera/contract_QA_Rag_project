import tensorflow_hub as hub
import numpy as np
import time

class ChatModelTester:
    def __init__(self, chat_qa):
        self.chat_qa = chat_qa
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")  # Load Universal Sentence Encoder

    def semantic_similarity(self, sentence1, sentence2):
        embeddings = self.use_model([sentence1, sentence2])
        similarity = np.inner(embeddings[0], embeddings[1])[0][0]
        return similarity

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
            
            # Calculate semantic similarity
            similarity_score = self.semantic_similarity(generated_answer, expected_answer)
            
            # Define a threshold for similarity score
            threshold = 0.8
            
            if similarity_score >= threshold:
                correct_answers += 1
                
            print(f"Question: {question}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Semantic Similarity Score: {similarity_score}")
            print(f"Response Time: {response_time} seconds\n")
        
        accuracy = (correct_answers / total_questions) * 100
        average_response_time = total_response_time / total_questions
        
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Response Time: {average_response_time:.2f} seconds")
