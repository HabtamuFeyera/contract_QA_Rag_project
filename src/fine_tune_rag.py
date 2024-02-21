import PyPDF2
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom PyPDFLoader class
class PyPDFLoader:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def load_text(self):
        text = ""
        with open(self.pdf_file, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            num_pages = reader.numPages
            for page_num in range(num_pages):
                page = reader.getPage(page_num)
                text += page.extractText()
        return text

# Define a custom dataset class
class ContractDataset(Dataset):
    def __init__(self, loaders, labels):
        self.loaders = loaders
        self.labels = labels

    def __len__(self):
        return len(self.loaders)

    def __getitem__(self, idx):
        text = self.loaders[idx].load_text()
        return text, self.labels[idx]

# Define the fine-tune function
def fine_tune_rag_model(train_dataset):
    # Load pre-trained RAG components
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_rag_model")

# Define the loaders
loaders = [
    PyPDFLoader("/home/habte/Downloads/Raptor Contract.docx.pdf"),
    PyPDFLoader("/home/habte/Downloads/Raptor Q&A2.docx.pdf"),
    PyPDFLoader("/home/habte/Downloads/Robinson Advisory.docx.pdf"),
    PyPDFLoader("/home/habte/Downloads/Robinson Q&A.docx.pdf")
]

# Example labels (replace with your actual labels)
labels = [0, 1, 2, 3]

# Create an instance of the ContractDataset
train_dataset = ContractDataset(loaders, labels)

# Fine-tune the RAG model
fine_tune_rag_model(train_dataset)
