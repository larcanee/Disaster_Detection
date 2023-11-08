import pandas as pd
import re
import nltk
import demoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

df = pd.read_csv('tweets.csv')  

nltk.download('stopwords')
nltk.download('wordnet')
demoji.download_codes()

# Initialize the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove links (URLs)
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text)

    # Remove emojis
    text = demoji.replace(text, repl="")  # Remove emojis using demoji

    # Remove words and numbers with digits
    text = re.sub(r'\w*\d\w*', '', text)

    # Eliminate stop words and lemmatize
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [wordnet_lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Remove special characters
    text = ' '.join(filtered_words)
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace special characters with spaces

    # Remove emoticons
    text = re.sub(r'(?::|;|\s)?(?:-)?(?:\)|D|P|3|\s|/)', ' ', text)

    # Remove non-ASCII characters
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)

    return text

# Handles null values in the 'keyword' column
def replace_null_keyword(df):
    df['keyword'].fillna('no_keyword', inplace=True)

# Handles null values in the 'location' column
def replace_null_location(df):
    df['location'].fillna('no_location', inplace=True)


df['preprocessed_text'] = df['text'].apply(preprocess_text)
replace_null_location(df)
replace_null_keyword(df)

df.to_csv('preprocessed_data.csv', index=False)

print("Data points count: ", df['id'].count())