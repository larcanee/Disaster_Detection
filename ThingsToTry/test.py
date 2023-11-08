import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import nltk, re, string
from string import punctuation
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
#machine learning
from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
# machine learning
from sklearn.naive_bayes import MultinomialNB,GaussianNB
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv('train.csv')
df.head()
df.info()

sns.set_style("dark")
sns.countplot(df.target)

# craeteing new column for storing length of reviews 
df['length'] = df['text'].apply(len)
df.head()

df['length'].plot(bins=50, kind='hist')

df.length.describe()

df[df['length'] == 157]['text'].iloc[0]

df.hist(column='length', by='target', bins=50,figsize=(10,4))