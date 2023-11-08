import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# df = pd.read_csv('preprocessed_data.csv') 

# plt.figure(figsize = (20,20)) # Text that is Disaster tweets
# wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(df.preprocessed_text))
# plt.imshow(wc , interpolation = 'bilinear')
# plt.axis("off")
# plt.show()

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

comment_words = ''

df = pd.read_csv('preprocessed_data.csv') 
df1 = df[df['target'] == 1]
df0 = df[df['target'] == 0]

df["wordcount"] = df.preprocessed_text.str.split().map(lambda x: len(x))
  
# iterate through the csv file
for val in df1.preprocessed_text:
    
    
    # typecaste each val to string
    val = str(val)
  
    # split the value
    tokens = val.split()
      
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
      
    comment_words += " ".join(tokens)+" "
  

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                min_font_size = 10,
                colormap='Set2', collocations=False).generate(comment_words) 

# plot the WordCloud image                        
plt.figure(figsize = (5, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Disaster Words")
plt.show() 

# iterate through the csv file
for val in df0.preprocessed_text:
    
    
    # typecaste each val to string
    val = str(val)
  
    # split the value
    tokens = val.split()
      
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
      
    comment_words += " ".join(tokens)+" "
  

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                min_font_size = 10,
                colormap='Set2', collocations=False).generate(comment_words) 

# plot the WordCloud image                        
plt.figure(figsize = (5, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Nondisaster Words")

plt.show() 

# most used disaster keywords
print(df1.keyword.nunique())

plt.figure(figsize = (9,6))
sns.countplot(y= df1.keyword, order = df1.keyword.value_counts().iloc[:10].index)
plt.title('10 Most Used Disaster Keywords')
plt.show()

# most used disaster keywords
print(df0.keyword.nunique())

plt.figure(figsize = (9,6))
sns.countplot(y= df0.keyword, order = df0.keyword.value_counts().iloc[:10].index)
plt.title('10 Most Used Nondisaster Keywords')
plt.show()