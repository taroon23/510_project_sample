import pandas as pd
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


amazon_data_file_path = '../amazon_data.csv'

amazon_headers = ['Brand', 'Date', 'Review']

# Read the CSV file into a pandas DataFrame with headers
amazon_data = pd.read_csv(amazon_data_file_path, names=amazon_headers)



google_data_file_path = '../google_data.csv'

google_headers = ['Brand', 'Date', 'Review']

# Read the CSV file into a pandas DataFrame with headers
google_data = pd.read_csv(google_data_file_path, names=google_headers)


tickers = ['ADDYY', 'NKE', 'SKX', 'UAA', 'PUM.DE']
brand_mapping = {'ADDYY': 'Adidas', 'NKE': 'Nike', 'SKX': 'Skechers', 'UAA': 'Under Armour', 'PUM.DE': 'Puma'}
stock_dfs = []

for ticker in tickers:
    #query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1514764800&period2=1713404238&interval=1d&events=history&includeAdjustedClose=true"
    
    query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=1464739200&period2=1714351190&interval=1d&events=history&includeAdjustedClose=true"
    
    stock_df = pd.read_csv(query_string)
    stock_df['Ticker'] = ticker
    stock_df['Brand'] = brand_mapping[ticker]
    stock_dfs.append(stock_df)

stock_data = pd.concat(stock_dfs, ignore_index=True)


amazon_google_data = pd.concat([amazon_data, google_data], ignore_index=True)

# Merge the two DataFrames on 'Brand' and 'Date'
combined_data = pd.merge(amazon_google_data, stock_data[['Date', 'Close', 'Brand']], on=['Date', 'Brand'], how='left')

stock_analysis_data = combined_data = combined_data.dropna(subset=['Close'])




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

print(stock_analysis_data)