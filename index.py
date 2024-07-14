from flask import Flask, send_from_directory,render_template,request
import nltk
nltk.download('vader_lexicon')
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/show_csv/<path:filename>')
def show_csv(filename):
    file_path = os.path.join('static', filename)
    df = pd.read_csv(file_path)
    table_html = df.head(100).to_html(classes='table table-striped')
    return render_template('display_csv.html', table_html=table_html)

@app.route('/show_csv2/<path:filename>')
def show_csv2(filename):
    file_path = os.path.join('static', filename)
    df = pd.read_csv(file_path)
    rowsDisplay = [1, 166, 850, 1914]
    finalRows = []
    for row in rowsDisplay:
        finalRows.extend(range(row, row + 21))
    df2 = df.iloc[finalRows]
    table_html = df2.to_html(classes='table table-striped', index=False)
    return render_template('display_csv.html', table_html=table_html)

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        polarity, subjectivity = analyze_sentiment_textblob(text)
        sentiment=analyze_sentiment_nltk(text)
        out=(polarity+sentiment)/2
        if(out<0) :   output="Negative"
        elif(out>0):   output="Positve"
        else     :       output="Neutral"
        return render_template('result.html', text=text, polarity=polarity, subjectivity=subjectivity,sentiment=sentiment,output=output)

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity
def analyze_sentiment_nltk(text):
    sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
    return sentiment['compound']

if __name__ == '__main__':
    app.run(debug=True,port=5001)
