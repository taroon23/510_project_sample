import pandas as pd
import streamlit as st

#pip install nltk

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Read the CSV files
amazon_data = pd.read_csv('amazon_data.csv', names=['Brand', 'Date', 'Review'])
google_data = pd.read_csv('google_data.csv', names=['Brand', 'Date', 'Review'])

# Read stock data
tickers = ['ADDYY', 'NKE', 'SKX', 'UAA', 'PUM.DE']
brand_mapping = {'ADDYY': 'Adidas', 'NKE': 'Nike', 'SKX': 'Skechers', 'UAA': 'Under Armour', 'PUM.DE': 'Puma'}
stock_dfs = []

for ticker in tickers:
    query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1464739200&period2=1714351190&interval=1d&events=history&includeAdjustedClose=true"
    stock_df = pd.read_csv(query_string)
    stock_df['Ticker'] = ticker
    stock_df['Brand'] = brand_mapping[ticker]
    stock_dfs.append(stock_df)

stock_data = pd.concat(stock_dfs, ignore_index=True)

# Merge data
amazon_google_data = pd.concat([amazon_data, google_data], ignore_index=True)
combined_data = pd.merge(amazon_google_data, stock_data[['Date', 'Close', 'Brand']], on=['Date', 'Brand'], how='left')
stock_analysis_data = combined_data.dropna(subset=['Close'])

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to assign sentiment labels using VADER
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Convert 'Review' column to string type
stock_analysis_data['Review'] = stock_analysis_data['Review'].astype(str)

# Apply sentiment analysis to each review
stock_analysis_data['Sentiment'] = stock_analysis_data['Review'].apply(get_sentiment)

# Streamlit app
st.title('Stock Sentiment Analysis')

# Display data
st.write(stock_analysis_data)
