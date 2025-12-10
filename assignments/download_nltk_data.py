import nltk

# Download all required NLTK data for the notebook
print("Downloading NLTK data packages...")

nltk.download('punkt')
nltk.download('punkt_tab')  # Required for word_tokenize in NLTK 3.9+
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual Wordnet for better lemmatization

print("\nAll NLTK data packages downloaded successfully!")
print("You can now run your notebook without errors.")
