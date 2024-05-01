import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    
    
    amazon_google_data['Date'] = pd.to_datetime(amazon_google_data['Date'])

    # Read the stock data
    tickers = ['ADDYY', 'NKE', 'SKX', 'UAA', 'PUM.DE']
    brand_mapping = {'ADDYY': 'Adidas', 'NKE': 'Nike', 'SKX': 'Skechers', 'UAA': 'Under Armour', 'PUM.DE': 'Puma'}
    stock_dfs = []
    for ticker in tickers:
        query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1464739200&period2=1714351190&interval=1d&events=history&includeAdjustedClose=true"
        stock_df = pd.read_csv(query_string)
        stock_df['Ticker'] = ticker
        stock_df['Brand'] = brand_mapping[ticker]
        # Convert 'Date' column to datetime
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_dfs.append(stock_df)
    stock_data = pd.concat(stock_dfs, ignore_index=True)

    # Merge stock data with review data
    combined_data = pd.merge(amazon_google_data, stock_data[['Date', 'Close', 'Brand']], on=['Date', 'Brand'], how='left')

    # Drop rows with missing values in the 'Close' column
    stock_analysis_data = combined_data.dropna(subset=['Close'])

    return stock_analysis_data, stock_data

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
stock_analysis_data, stock_data = load_data()

# Perform sentiment analysis
stock_analysis_data = perform_sentiment_analysis(stock_analysis_data)

# Streamlit app
def main():
    st.title('Stock Analysis Data App')

    # Page navigation
    page = st.sidebar.radio("Navigate", ['Start Page', 'Analysis Page', 'Brand Analysis Page'])

    if page == 'Start Page':
        st.header('Start Page')

    elif page == 'Analysis Page':
        st.header('Analysis Page')
        st.write(stock_analysis_data)

    elif page == 'Brand Analysis Page':
        st.header('Brand Analysis Page')

        # Dropdown for selecting brand
        selected_brand = st.selectbox("Select Brand", stock_analysis_data['Brand'].unique())

        # Filter data for selected brand
        selected_brand_data = stock_analysis_data[stock_analysis_data['Brand'] == selected_brand]

        # Dropdown for selecting year
        years = selected_brand_data['Date'].dt.year.unique()
        years = np.append('All', years)
        selected_year = st.selectbox("Select Year", years)

        # Filter data for selected year
        if selected_year != 'All':
            selected_brand_data = selected_brand_data[selected_brand_data['Date'].dt.year == selected_year]

        # Disable the warning
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Display pie chart and line graph side by side
        col1, col2 = st.columns([1, 1])

        # Pie chart for sentiment analysis
        with col1:
            sentiment_counts = selected_brand_data['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.subheader(f"Sentiment Analysis for {selected_brand}")
            st.pyplot(fig)

        # Line graph for stock value across dates
        with col2:
            if selected_year != 'All':
                selected_stock_data = stock_data[(stock_data['Brand'] == selected_brand) & (stock_data['Date'].dt.year == selected_year)]
            else:
                selected_stock_data = stock_data[stock_data['Brand'] == selected_brand]
            selected_stock_data = selected_stock_data.sort_values(by='Date')
            fig, ax = plt.subplots()
            sns.lineplot(data=selected_stock_data, x='Date', y='Close')
            st.subheader(f"Stock Value for {selected_brand}")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
