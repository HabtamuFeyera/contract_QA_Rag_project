from transformers import RagTokenizer, RagTokenForGeneration, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.questions[idx], return_tensors="pt", padding="max_length", truncation=True)
        labels = self.tokenizer.encode(self.answers[idx], return_tensors="pt", padding="max_length", truncation=True).squeeze()
        return inputs, labels

# Load the RAG tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

# Define the training data 
questions = ["What is the capital of France?", "Who wrote Hamlet?"]
answers = ["Paris", "William Shakespeare"]

# Create an instance of the QADataset
train_dataset = QADataset(questions, answers, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Instantiate the generator component of the RAG model
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the generator component
trainer.train()

# Save the trained generator model
model.save_pretrained("fine_tuned_generator")
