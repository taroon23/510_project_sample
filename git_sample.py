import pandas as pd
import streamlit as st

#pip install nltk

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Function to load data
def load_data():
    # Read the data files
    amazon_data = pd.read_csv('amazon_data.csv', names=['Brand', 'Date', 'Review'])
    google_data = pd.read_csv('google_data.csv', names=['Brand', 'Date', 'Review'])

    # Concatenate Amazon and Google data
    amazon_google_data = pd.concat([amazon_data, google_data], ignore_index=True)

    # Read the stock data
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

    # Merge stock data with review data
    combined_data = pd.merge(amazon_google_data, stock_data[['Date', 'Close', 'Brand']], on=['Date', 'Brand'], how='left')

    # Drop rows with missing values in the 'Close' column
    stock_analysis_data = combined_data.dropna(subset=['Close'])

    return stock_analysis_data

# Function to perform sentiment analysis
def perform_sentiment_analysis(data):
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
    data['Review'] = data['Review'].astype(str)

    # Apply sentiment analysis to each review
    data['Sentiment'] = data['Review'].apply(get_sentiment)

    return data

# Load the data
stock_analysis_data = load_data()

# Perform sentiment analysis
stock_analysis_data = perform_sentiment_analysis(stock_analysis_data)

# Streamlit app
def main():
    st.title('Stock Analysis Data App')

    # Page navigation
    page = st.sidebar.radio("Navigate", ['Start Page', 'Analysis Page'])

    if page == 'Start Page':
        st.header('Start Page')
        #st.write("Click the button below to navigate to the Analysis Page.")
        #if st.button("Go to Analysis Page"):
            #st.experimental_set_query_params(page='Analysis Page')

    elif page == 'Analysis Page':
        st.header('Analysis Page')
        st.write(stock_analysis_data)

if __name__ == "__main__":
    main()