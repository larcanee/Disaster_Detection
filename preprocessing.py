import pandas as pd
import re
import nltk
import demoji
from nltk.corpus import stopwords

# Read the CSV file into a DataFrame
df = pd.read_csv('sample.csv')  # Replace 'your_data.csv' with your file path
nltk.download('stopwords')

# Download demoji's emoji database
demoji.download_codes()

# Define the preprocessing function
def preprocess_text(text):
    # Step 1: Convert text to lowercase
    text = text.lower()

    # Step 2: Remove links (URLs)
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text)

    # Step 3: Remove emojis
    text = demoji.replace(text, repl="")  # Remove emojis using demoji

    # Step 4: Remove words and numbers with digits
    text = re.sub(r'\w*\d\w*', '', text)

    # Step 5: Eliminate stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    # Step 6: Remove special characters
    text = ' '.join(filtered_words)
    text = re.sub(r'[^\w\s]', '', text)

    # Step 7: Remove non-ASCII characters
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)

    return text

# Apply preprocessing to each entry in a specific column (e.g., 'text_column')
df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)  # Save to a new file, replace 'preprocessed_data.csv' with your desired filename

# input_text = "Computer deciphers lower case and capitalized letters differently, and game57 is a challenge. http://example.com"
# preprocessed_text = preprocess_text(input_text)
# print(preprocessed_text)