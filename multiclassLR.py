import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your CSV file
csv_file_path = "preprocessed_data.csv"
df = pd.read_csv(csv_file_path)

# Assuming your CSV file has columns for features and a column for labels
X_text = df['preprocessed_text']  # Replace "feature_column" with the actual column containing text data
y = df['target']  # Replace "target" with the actual label column name

# Encode categorical labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# # Tokenize and pad the text data
# max_words = 5000
# max_len = 100

# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(X_train_text)

# X_train_seq = tokenizer.texts_to_sequences(X_train_text)
# X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
# X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build and train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train_text, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_text)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
