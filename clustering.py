import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file with tweet data
file_path = 'output_csv_files/tweets_label_1.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Assuming you have a 'preprocessed_text' column containing the tweet text

# Extract tweet text
tweets = df['preprocessed_text'].astype(str)

# Use TF-IDF Vectorizer to convert text data to numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(tweets)

# Apply K-Means clustering
num_clusters = 2  # You can adjust the number of clusters based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Calculate silhouette score
silhouette_avg = silhouette_score(X, df['cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the clusters (assuming 2D data for simplicity)
# For simplicity, I'm using the first two columns of the TF-IDF vectors for the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0].toarray().flatten(), y=X[:, 1].toarray().flatten(), hue='cluster', data=df, palette='Set1', legend='full')
plt.title('Tweet Clustering')
plt.show()
