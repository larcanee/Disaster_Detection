import tweepy

# Set up Twitter API credentials
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create a streaming listener
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        # Process and classify the tweet using your model
        tweet_text = status.text
        classification_result = classify_tweet(tweet_text)
        print(f"Tweet: {tweet_text}\nClassification: {classification_result}")

# Create a streaming API object
my_stream_listener = MyStreamListener()
my_stream = tweepy.Stream(auth=auth, listener=my_stream_listener)

# Start streaming with specific keywords
my_stream.filter(track=["disaster", "emergency", "crisis"])
