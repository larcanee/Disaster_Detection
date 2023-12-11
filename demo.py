import pandas as pd
import seaborn as sns
import re
import nltk
import demoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection,naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import os

# PREPROCESSING
df_phase1 = pd.read_csv('train.csv')
df_phase2 = pd.read_csv('categorynums.csv') 

nltk.download('stopwords')
nltk.download('wordnet')
demoji.download_codes()

wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # convert text to lowercase
    text = text.lower()

    # remove links
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text)

    # remove user mentions 
    text = re.sub(r'@\w+', '', text)

    # remove emojis
    text = demoji.replace(text, repl="")

    # remove words with digits
    text = re.sub(r'\w*\d\w*', '', text)

    # eliminate stop words and lemmatize
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [wordnet_lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # remove special characters
    text = ' '.join(filtered_words)
    text = re.sub(r'[^\w\s]', '', text)

    # remove emoticons
    text = re.sub(r'(?::|;|\s)?(?:-)?(?:\)|D|P|3|\s|/)', ' ', text)

    # remove non-ASCII characters
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)

    return text

def preprocess_keyword(keyword):
    # Replace numbers with an empty string
    cleaned_keyword = re.sub(r'\d', '', str(keyword))
    
    # Remove non-alphabetic characters, convert to lowercase, and add a space between words
    cleaned_keyword = re.sub(r'[^a-zA-Z\s%]', '', cleaned_keyword).lower()
    
    # Replace a percent sign with a space
    cleaned_keyword = cleaned_keyword.replace('%', ' ')
    
    return cleaned_keyword

def replace_null_keyword(df):
    df['keyword'] = df['keyword'].replace(['nan', pd.NaT], 'no_keyword')

def replace_null_location(df):
    df['location'].fillna('no_location', inplace=True)


df_phase1['preprocessed_text'] = df_phase1['text'].apply(preprocess_text)
df_phase2['preprocessed_text'] = df_phase2['text'].apply(preprocess_text)
replace_null_location(df_phase1)
df_phase1['keyword'] = df_phase1['keyword'].apply(preprocess_keyword)
replace_null_keyword(df_phase1)

df_phase1.to_csv('preprocessed_data.csv', index=False)
df_phase2.to_csv('categories_preprocessed.csv', index=False)

print("Data points count: ", df_phase1['id'].count())
print("Data points count: ", df_phase2['id'].count())

target_df_phase1 = df_phase1.target.value_counts().reset_index()
target_df_phase1.columns = ['target', 'count']
fig = px.pie(target_df_phase1, values='count', names='target', title='Target Classification',
             color_discrete_sequence=['red', 'light blue'])
# fig.show()

target_df_phase2 = df_phase2.target.value_counts().reset_index()
target_df_phase2.columns = ['target', 'count']
fig = px.pie(target_df_phase2, values='count', names='target', title='Target Classification',
             color_discrete_sequence=['red', 'light blue'])
# fig.show()


# LOGISTIC REGRESSION
df_phase1 = pd.read_csv('preprocessed_data.csv') 
df_phase1['preprocessed_text'].fillna('', inplace=True)

# Separate features and labels
X = df_phase1['preprocessed_text']
y = df_phase1['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression Model
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test_tfidf)

# Evaluate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for Phase 1: {:.2f}".format(accuracy))

# Print the classification report
print('Classification Report for Phase 1:\n', classification_report(y_test, y_pred))

# filter and save the data classified as 1 to a new CSV file
df_classified_as_1 = df_phase1[df_phase1['target'] == 1]
df_classified_as_1.to_csv('disaster_tweets.csv', index=False)

# MULTICLASS LOGISTIC REGRESSION
df_train = pd.read_csv('categoriespreprocessed.csv')

# Separate features and labels for training data
X_train = df_train['preprocessed_text']
y_train = df_train['target']

# Encode labels into numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Create a pipeline with TF-IDF vectorizer and logistic regression model
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),  # Adjust max_features as needed
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42))
])

# Train the model using the pipeline
model.fit(X_train, y_train_encoded)

# Load the testing data
df_test = pd.read_csv('disaster_tweets.csv')

# Separate features for test data
X_test = df_test['text']

# Make predictions on the test set
y_pred = model.predict(X_test)

# Output the predictions
df_test['predicted_label'] = label_encoder.inverse_transform(y_pred)
df_test.to_csv('predictions.csv', index=False)

label_array = label_encoder.classes_  # Use label classes from training data

# Create a directory to store individual CSV files
output_directory = "output_csv_files"
os.makedirs(output_directory, exist_ok=True)

for label in label_array:
    # Filter the test DataFrame based on the label
    filtered_df = df_test[df_test['predicted_label'] == label]

    # Define the filename based on the label
    output_filename = f"{output_directory}/tweets_label_{label}.csv"

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_filename, index=False, encoding="utf-8")

    print(f"Tweets with label '{label}' exported to {output_filename}")