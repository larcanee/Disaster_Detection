import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection,naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('preprocessed_data.csv') 
df['preprocessed_text'].fillna('', inplace=True)

df_encoded = df.copy(deep = True)
le_gen = LabelEncoder()
df_encoded['keyword'] = le_gen.fit_transform(df['keyword'])

Train_X,Test_X,Train_Y,Test_Y = model_selection.train_test_split(df['preprocessed_text'],df['target'],test_size = 0.3)
encoder = LabelEncoder()
Train_Y = encoder.fit_transform(Train_Y)
Test_Y = encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features = 5000)
Tfidf_vect.fit(df['preprocessed_text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

logreg = LogisticRegression(solver='liblinear')
logreg.fit(Train_X_Tfidf,Train_Y)

predictions = logreg.predict(Test_X_Tfidf)
print(predictions)
print("Accuracy : ", accuracy_score(predictions,Test_Y))