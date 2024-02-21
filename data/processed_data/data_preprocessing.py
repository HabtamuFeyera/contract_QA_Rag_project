import PyPDF2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class PyPDFLoader:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def load_text(self):
        text = ""
        with open(self.pdf_file, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extractText()
        return text

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

# List of PDF file paths
pdf_files = [
    "/home/habte/Downloads/Raptor Contract.docx.pdf",
    "/home/habte/Downloads/Raptor Q&A2.docx.pdf",
    "/home/habte/Downloads/Robinson Advisory.docx.pdf",
    "/home/habte/Downloads/Robinson Q&A.docx.pdf"
]

# Load text from PDF files and preprocess
processed_texts = []
for file_path in pdf_files:
    pdf_loader = PyPDFLoader(file_path)
    text = pdf_loader.load_text()
    processed_text = preprocess_text(text)
    processed_texts.append(processed_text)

# Print processed texts
for i, text in enumerate(processed_texts):
    print(f"Processed Text {i+1}: {text}")
