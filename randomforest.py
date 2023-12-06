import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load your preprocessed data
# Assuming you have a CSV file with columns 'text' and 'label'
df = pd.read_csv('preprocessed_data.csv')

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
features = tfidf_vectorizer.fit_transform(df['text']).toarray()
labels = df['target']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Building the Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Filter and save the data classified as 1 to a new CSV file
# df_classified_as_1 = df[df['target'] == 1]
# df_classified_as_1.to_csv('disaster_tweets.csv', index=False)