import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load your preprocessed data
# Assuming you have a CSV file with columns 'text' and 'label'
df = pd.read_csv('preprocessed_data.csv')
df['preprocessed_text'].fillna('', inplace=True)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
features = tfidf_vectorizer.fit_transform(df['preprocessed_text']).toarray()
labels = df['target']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Building the Multinomial Naive Bayes Model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
