import os
import pandas as pd
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def analyze_sentiment_nltk(text):
    sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
    return sentiment['compound']

def analyze_and_save_textblob_tweet(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Polarity'], df['Subjectivity'] = zip(*df['Tweet'].apply(analyze_sentiment_textblob))
    df.to_csv(output_file, index=False)

def analyze_and_save_nltk_tweet(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Sentiment'] = df['Tweet'].apply(analyze_sentiment_nltk)
    df.to_csv(output_file, index=False)

def analyze_and_save_textblob_yt(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Polarity'], df['Subjectivity'] = zip(*df['Description'].apply(analyze_sentiment_textblob))
    df.to_csv(output_file, index=False)

def analyze_and_save_nltk_yt(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Sentiment'] = df['Description'].apply(analyze_sentiment_nltk)
    df.to_csv(output_file, index=False)


input_tweet_csv = os.path.join('static', 'InputDatatweet.csv')
output_tweet_textblob_csv = os.path.join('static', 'tweet_textblob.csv')
output_tweet_nltk_csv = os.path.join('static', 'tweet_nltk.csv')
input_yt_csv = os.path.join('static', 'InputDataYT.csv')
output_yt_textblob_csv = os.path.join('static', 'yt_textblob.csv')
output_yt_nltk_csv = os.path.join('static', 'yt_nltk.csv')

analyze_and_save_textblob_tweet(input_tweet_csv, output_tweet_textblob_csv)
analyze_and_save_textblob_yt(input_yt_csv, output_yt_textblob_csv)
analyze_and_save_nltk_tweet(input_tweet_csv, output_tweet_nltk_csv)
analyze_and_save_nltk_yt(input_yt_csv, output_yt_nltk_csv)
