import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_contract_document(contract_text):
    # Tokenization
    tokens = word_tokenize(contract_text)
    
    # Remove punctuation and stopwords
    tokens = [token for token in tokens if token not in string.punctuation and token not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Convert tokens back to text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Example contract document
contract_text = "This agreement is entered into by and between Company X and Company Y"
processed_contract_text = preprocess_contract_document(contract_text)
print(processed_contract_text)
