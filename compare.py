import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your preprocessed tweets data from the CSV file
# Replace 'your_dataset.csv' with the actual path to your CSV file
csv_file_path = 'categories.csv'
df = pd.read_csv(csv_file_path)

# Assuming your CSV has a 'text' column for tweets and a 'label' column for class labels
X = df['text']
y = df['target']

# Convert class labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert tweet text to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define a list of classifiers
classifiers = [
    # LogisticRegression(max_iter=1000),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(),
    # SVC(),
    # KNeighborsClassifier(),
    # MultinomialNB(),
    # MLPClassifier(max_iter=1000),
     GradientBoostingClassifier()
]

# Create a list to store the results
results_list = []

# Iterate over classifiers and evaluate performance
for clf in classifiers:
    clf_name = clf.__class__.__name__
    clf.fit(X_train_vectorized, y_train)
    y_pred = clf.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    results_list.append({'Classifier': clf_name, 'Accuracy': accuracy})

# Convert the list of results to a DataFrame
results_df = pd.DataFrame(results_list)

# Print the results
print(results_df)

# Print detailed classification report for each classifier
for clf in classifiers:
    clf_name = clf.__class__.__name__
    clf.fit(X_train_vectorized, y_train)
    y_pred = clf.predict(X_test_vectorized)
    print(f"\nClassification Report for {clf_name}:\n")
    print(classification_report(y_test, y_pred))
