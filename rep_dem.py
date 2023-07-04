#!/usr/bin/env python
# coding: utf-8

# # Democrates Vs Republicans 
# # Using NLP

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')

import re
from nltk.stem import PorterStemmer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from wordcloud import WordCloud,STOPWORDS
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings('ignore')


# In[8]:


data = pd.read_csv('tweet.csv')
data


# In[9]:


data.rename(columns={"Handle": "Politician"}, inplace=True)


# In[10]:


#number of each party tweets:
data['Party'].value_counts()


# In[11]:


#number of politicians in the dataset:
data['Politician'].value_counts()


# # NLP
# # Sentiment Analysis

# ## Preprocessing

# In[12]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\d+", "", tweet)
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Tokenization
    tokens = word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    processed_tweet = " ".join(tokens)
    
    return processed_tweet

# Preprocess the 'Tweet' column
data['ProcessedTweet'] = data['Tweet'].apply(preprocess_tweet)

data


# ## Using SentimentIntensityAnalyzer

# In[13]:


sia = SentimentIntensityAnalyzer()
res = {}
for i in range(len(data)):
    tweet = data['ProcessedTweet'][i]
    politician = data['Politician'][i]
    res[politician] = sia.polarity_scores(tweet)

df = pd.DataFrame(res).T
df


# In[14]:


df = df.reset_index().rename(columns={'index': 'Politician'})
data = data.merge(df, how='right')
data.drop(['neg', 'neu', 'pos'], axis=1, inplace=True)


# In[15]:


data


# ##  Using textblob

# In[16]:


from textblob import TextBlob


# In[17]:


# Function to perform sentiment analysis using TextBlob
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return polarity, sentiment

# Apply sentiment analysis to the "ProcessedTweet" column and store results in new columns
data[['Compound_TextBlob', 'Sentiment_TextBlob']] = data['ProcessedTweet'].apply(get_sentiment).apply(pd.Series)


# In[18]:


data


# In[19]:


data["ProcessedTweet"].loc[86458]


# In[20]:


#it looks like textblob is doing a better job!


# In[21]:


#Label Encoding in case of using classifiers


# In[22]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the Predicted Sentiment
data['Encoded_txtblb'] = label_encoder.fit_transform(data['Sentiment_TextBlob'])

data


# In[23]:


data['Encoded_txtblb'].value_counts()


# ## Positivity Average for each politician

# In[26]:


df = data.groupby('Politician')['Compound_TextBlob'].mean().reset_index()
df


# ## demonstrating politicians with most and least average

# In[35]:


# Sort the dataframe by 'Average_Compound_TextBlob' in descending order for top politicians
top_10_politicians = df.nlargest(10, 'Compound_TextBlob')

# Sort the dataframe by 'Average_Compound_TextBlob' in ascending order for bottom politicians
bottom_10_politicians = df.nsmallest(10, 'Compound_TextBlob')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# Plot the top 10 politicians in the first subplot
ax1.bar(top_10_politicians['Politician'], top_10_politicians['Compound_TextBlob'], color='skyblue',
        edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Politician', fontsize=12)
ax1.set_ylabel('Average Compound_TextBlob', fontsize=12)
ax1.set_title('Top 10 Politicians with Highest Compound_TextBlob', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=90)

# Plot the bottom 10 politicians in the second subplot
ax2.bar(bottom_10_politicians['Politician'], bottom_10_politicians['Compound_TextBlob'], color='salmon',
        edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Politician', fontsize=12)
ax2.set_ylabel('Average Compound_TextBlob', fontsize=12)
ax2.set_title('Bottom 10 Politicians with Lowest Compound_TextBlob', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=90)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# ## Positivity Average for each party

# In[28]:


df1 = data.groupby('Party')['Compound_TextBlob'].mean().reset_index()
df1


# In[29]:


# Define the bar width
bar_width = 0.35

# Define the colors
colors = ['#779ECB', '#D98880']

# Create the bar plot
plt.bar(df1['Party'], df1['Compound_TextBlob'], width=bar_width, color=colors)
plt.xlabel('Party')
plt.ylabel('Compound_TextBlob')
plt.title('Compound_TextBlob by Party')

# Adjust the spacing between the bars
plt.subplots_adjust(wspace=0.1)

# Display the plot
plt.show()


# ## Frequency of Usage of Words by Parties

# In[30]:


from nltk.probability import FreqDist
fdist_democrat = FreqDist(data[data["Party"] == "Republican"])
fdist_republican=FreqDist(data[data["Party"] == "Democrat"])


# ## Word Clouds

# In[31]:


from PIL import Image


# In[32]:


# Open the image file
image_path = "us.png"
image = Image.open(image_path)

# Convert the image to grayscale
image_gray = image.convert("L")

# Convert the image to an array of pixel values
image_array = np.array(image_gray)

# Threshold the image to create a mask
threshold = 150  # Adjust the threshold value as needed
mask = np.where(image_array > threshold, 255, 0).astype(np.uint8)

# Filter the data for Republican tweets
republican_tweets = data[data["Party"] == "Republican"]

# Get the text from the "ProcessedTweet" column for Republican tweets
text = " ".join(republican_tweets["ProcessedTweet"].values)

# Remove "rt" and "amp" from the word cloud
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["rt", "amp", "u", "thank", "today", "new", "thanks", "W"])

# Generate the word cloud with mask parameter and adjusted canvas size
wordcloud = WordCloud(
    background_color="white",
    mask=mask,
    contour_width=1,
    contour_color="black",
    width=1000,  # Increase the canvas width
    height=1000,  # Increase the canvas height
    stopwords=custom_stopwords,  # Use custom stopwords
    colormap="Reds"  # Set the colormap to Reds for red color style
).generate(text)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(wordcloud, interpolation="bilinear")
ax.set_axis_off()

# Display the image with the word cloud
plt.imshow(wordcloud, cmap=plt.cm.Reds, interpolation="bilinear")
plt.axis("off")
plt.title("REPUBLICANS")
plt.savefig("rep_wc.png")
plt.show()


# In[33]:


# Open the image file
image_path = "us.png"
image = Image.open(image_path)

# Convert the image to grayscale
image_gray = image.convert("L")

# Convert the image to an array of pixel values
image_array = np.array(image_gray)

# Threshold the image to create a mask
threshold = 150  # Adjust the threshold value as needed
mask = np.where(image_array > threshold, 255, 0).astype(np.uint8)

# Filter the data for Democrat tweets
democrat_tweets = data[data["Party"] == "Democrat"]

# Get the text from the "ProcessedTweet" column for Democrat tweets
text = " ".join(democrat_tweets["ProcessedTweet"].values)

# Remove "rt" and "amp" from the word cloud
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["rt", "amp", "u", "thank", "today", "new"])

# Generate the word cloud with mask parameter and adjusted canvas size
wordcloud = WordCloud(
    background_color="white",
    mask=mask,
    contour_width=1,
    contour_color="black",
    width=1000,  # Increase the canvas width
    height=1000,  # Increase the canvas height
    stopwords=custom_stopwords,  # Use custom stopwords
    colormap="Blues"  # Set the colormap to Blues for blue color style
).generate(text)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(wordcloud, interpolation="bilinear")
ax.set_axis_off()

# Display the image with the word cloud
plt.imshow(wordcloud, cmap=plt.cm.Blues, interpolation="bilinear")
plt.axis("off")
plt.title("DEMOCRATS")
plt.savefig("Dem_wc.png")
plt.show()


# # Looks Like Democrats care more about us (students)! JK I'm not taking any sides:D
