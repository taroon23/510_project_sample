import pandas as pd
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib.dates import MonthLocator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet 
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud

import os
import serpapi

import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import time

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#Scraping Amazon Data
def get_amazon_data_function():
    amazon_request_data = {
        "authority": "www.amazon.com",
        "method": "GET",
        "path": "/s?k=shoes&crid=2BK2NPYYMOA0Y&sprefix=shoes%2Caps%2C139&ref=nb_sb_noss_1",
        "scheme": "https",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": "session-id=142-2984597-2384856; session-id-time=2082787201l; i18n-prefs=USD; ubid-main=130-4009680-2230565; session-token=H3InKS3swF5MqPMQVkPBFyVppkwfhvbeDwX3XjfISxUN5UX+pVQCYPIKV8AygjfSQqb29gs16H27541vpvu/EyjS+ikwstm2jvhY32F98M/aEbZsU+jgKQFr2gBf3Z4BPwAKTsB+bh0wjMkjhpXpiHLKNXj4kvSAQj1iheuDsPzYqzz+LTU7LbyOtF8AjykDB1ZlizUwsIkidW6yMbRAo+JfbR2JcVgMfDPA2v5rkkWJPMNd+uK/Xk4BZ/8WFK1se9+aoB9cAzIQexlfNNPLD3khcjUNI0SwoDkzalaekZa+WmmOmSo9iTXochPZjRDvIF1T3NYNqGAu22G+sSLsUFp0ye0gcQEr; JSESSIONID=E378761678CC34E66F03CBD470884431; csm-hit=tb:s-33FGFH4QX9293JJ11SY3|1712629669705&t:1712629673488&adb:adblk_no",
        "Device-Memory": "8",
        "Downlink": "10",
        "Dpr": "1.25",
        "Ect": "4g",
        "Rtt": "50",
        "Sec-Ch-Device-Memory": "8",
        "Sec-Ch-Dpr": "1.25",
        "Sec-Ch-Ua": "\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "\"Windows\"",
        "Sec-Ch-Ua-Platform-Version": "\"15.0.0\"",
        "Sec-Ch-Viewport-Width": "682",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Viewport-Width": "682"
    }

    def amazon_scrape_page(url):
        response = requests.get(url, headers=amazon_request_data)
        soup = BeautifulSoup(response.content, 'html.parser')

        with open('amazon_review_websites.csv', 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Brand', 'Shoe Name', 'Review Website']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            div_elements = soup.find_all('div', class_='a-section a-spacing-small puis-padding-left-micro puis-padding-right-micro')
            for div_element in div_elements:
                specific_brand_element = div_element.find('span', class_='a-size-base-plus a-color-base')
                brand = specific_brand_element.get_text()        

                specific_shoe_element = div_element.find('span', class_='a-size-base-plus a-color-base a-text-normal')
                shoe_name = specific_shoe_element.get_text()

                specific_review_element = div_element.find('a', class_='a-link-normal s-underline-text s-underline-link-text s-link-style')
                if specific_review_element:
                    shoe_review_link = 'https://www.amazon.com/' + specific_review_element['href']
                else:
                    shoe_review_link = ''  # or any other default value
              
                writer.writerow({'Brand': brand, 'Shoe Name': shoe_name, 'Review Website': shoe_review_link})

    def amazon_get_next_page_url(url):
        response = requests.get(url, headers=amazon_request_data)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        next_page_element = soup.find('div', class_='a-section a-text-center s-pagination-container')
        if next_page_element:
            next_page_link_element = next_page_element.find('a')
            next_page_link = 'https://www.amazon.com/' + next_page_link_element.get('href')
            return next_page_link
        else:
            return None

    def amazon_scrape_multiple_pages(current_url,num_pages):
        current_url = amazon_base_url
        for _ in range(num_pages):
            amazon_scrape_page(current_url)
            next_page_url = amazon_get_next_page_url(current_url)
            if next_page_url:
                current_url = next_page_url
            else:
                break
            time.sleep(2)

    amazon_base_url = "https://www.amazon.com/s?k=shoes&i=fashion&rh=n%3A7141123011%2Cp_123%3A198664%7C234394%7C234502%7C256097%7C6832&dc&crid=PYFEZT5KCGKW&qid=1713488827&rnid=85457740011&sprefix=shoes%2Caps%2C165&ref=sr_nr_p_123_18&ds=v1%3AoZI0FyyIC2VbanXgXbIcp5RsGpGpkJmAIgcp2rdeV2E"
    amazon_num_pages = 5

    amazon_scrape_multiple_pages(amazon_base_url, amazon_num_pages)

    amazon_review_website_path = 'amazon_review_websites.csv'
    amazon_review_website_data = pd.read_csv(amazon_review_website_path)

    amazon_review_website_data = amazon_review_website_data.drop(columns=['Shoe Name'])
    amazon_review_website_data['Brand'] = amazon_review_website_data['Brand'].replace({'PUMA GOLF': 'PUMA', 'adidas': 'Adidas'})

    amazon_brands = amazon_review_website_data['Brand'].unique().tolist()
    amazon_brand_data = {}
    amazon_review_websites_by_brand = {}

    for brand_name in amazon_brands:
        amazon_brand_data[brand_name] = amazon_review_website_data[amazon_review_website_data['Brand'] == brand_name]
        
    for brand_name, brand_df in amazon_brand_data.items():
        amazon_review_websites = brand_df['Review Website'].tolist()
        amazon_review_websites_by_brand[brand_name] = amazon_review_websites

    def amazon_scrape_review(brand, url, id_initiator):
        shoe_ID = id_initiator
        review_response = requests.get(url, headers=amazon_request_data)
        soup = BeautifulSoup(review_response.content, 'html.parser')
        
        ratings = soup.find('span', class_='a-size-base a-color-base').text

        price = soup.find('span', class_='a-offscreen')
        if price:
            price_text = price.text.strip()  
            price_without_dollar = price_text.replace('$', '')
            if price_without_dollar:  
                price_float = float(price_without_dollar)
            else:
                price_float = np.nan
        else:
            price_float = np.nan

        review_list_div = soup.find('div', id='cm-cr-dp-review-list')
        
        if review_list_div:
            review_divs = review_list_div.find_all('div', class_='a-section review aok-relative')
            for review_div in review_divs:
                date_stuff_div = review_div.find('span', class_='a-size-base a-color-secondary review-date').text
                date_str = date_stuff_div.split("on ")[1]
                parsed_date = datetime.strptime(date_str, "%B %d, %Y")
                formatted_date = parsed_date.strftime("%Y-%m-%d")
                
                review_text_div = review_div.find('div', class_='a-expander-content reviewText review-text-content a-expander-partial-collapse-content')
                if review_text_div:
                    review = review_text_div.text.strip()
                    with open('amazon_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
                        fieldnames = ['Brand Name', 'Shoe_ID', 'Price', 'Date', 'Ratings', 'Review']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({'Brand Name': brand, 'Shoe_ID': shoe_ID, 'Price': price_float, 'Date': formatted_date, 'Ratings': ratings, 'Review': review})
                else:
                    print("No review text found.")
    
    id_initiator = 1

    for brand_name in amazon_brands:
        for url in amazon_review_websites_by_brand[brand_name]:
            if id_initiator > 240:
                break  # Break out of the loop if id_initiator exceeds 240 for now
            amazon_scrape_review(brand_name, url, id_initiator)
            id_initiator = id_initiator + 1

#Function to get data using Google Shopping API
def get_google_data_function():
    google_api_key = "c5834b9bf92b1098a81b551eac79a501bf105e1c831699cb4058ffd97376d51d"
    google_client = serpapi.Client(api_key= google_api_key)

    google_result = google_client.search({
        'engine': 'google_shopping',
        'q': 'men shoes',
        'tbs': 'mr:1,pdtr0:871889%7C872459!872292!871894!872392!872337'
    })

    google_review_website = [(item["title"], item["product_link"], item["extracted_price"]) for item in google_result["shopping_results"]]

    google_review_website_data = pd.DataFrame(google_review_website, columns=["title", "link", "price"])

    google_review_website_data['link'] = google_review_website_data['link'].str.replace('?gl=us', '/reviews?gl=us')

    keywords = ['Under Armour', 'Skechers', 'Puma', 'Adidas', 'Nike']

    google_review_website_data['Brand'] = np.select(
        [google_review_website_data['title'].str.contains(keyword) for keyword in keywords],
        keywords,
        default='NNN'
    )

    google_review_website_data = google_review_website_data[google_review_website_data['Brand'] != 'NNN']
    google_request_info = {
        "authority": "www.google.com",
        "method": "GET",
        "path": "/shopping/product/14880183997873103222/reviews?gl=us",
        "scheme": "https",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "Sec-Ch-Ua-Arch": "x86",
        "Sec-Ch-Ua-Bitness": '"64"',
        "Sec-Ch-Ua-Full-Version-List": '"Google Chrome";v="123.0.6312.122", "Not:A-Brand";v="8.0.0.0", "Chromium";v="123.0.6312.122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Model": '""',
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Ch-Ua-Platform-Version": '"15.0.0"',
        "Sec-Ch-Ua-Wow64": "?0",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }

    google_brands = ['Under Armour', 'Skechers', 'Puma', 'Adidas', 'Nike']

    google_brand_data = {}
    google_review_data_by_brand = {}

    for brand_name in google_brands:
        google_brand_data[brand_name] = google_review_website_data[google_review_website_data['Brand'] == brand_name]

    for brand_name, brand_df in google_brand_data.items():
        review_data = {'link': brand_df['link'].tolist(), 'price': brand_df['price'].tolist()}
        google_review_data_by_brand[brand_name] = review_data

    def google_scrape_review(brand, url, price, id_initiator):
        price_without_dollar = price
        price_float = float(price_without_dollar)

        shoe_ID = id_initiator

        review_response = requests.get(url, headers=google_request_info)
        soup = BeautifulSoup(review_response.content, 'html.parser')

        ratings_div = soup.find('div', class_='uYNZm')

        if ratings_div:  
            ratings = ratings_div.text.strip()
        else:
            ratings = np.nan

        review_divs = soup.find_all('div', class_='z6XoBf fade-in-animate')

        for review_div in review_divs:
            review_text_div = review_div.find('div', class_='g1lvWe')

            if review_text_div:
                date_stuff_div = review_div.find('span', class_='less-spaced ff3bE nMkOOb').text

                parsed_date = datetime.strptime(date_stuff_div, "%B %d, %Y")

                formatted_date = parsed_date.strftime("%Y-%m-%d")

                review = review_text_div.text.strip()

                with open('google_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Brand Name', 'Shoe_ID', 'Price', 'Date', 'Ratings', 'Review']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'Brand Name': brand, 'Shoe_ID': shoe_ID, 'Price': price_float, 'Date': formatted_date, 'Ratings': ratings, 'Review': review})

            else:
                print("No review text found.")

    id_initiator = 1

    for brand_name in google_brands:
        review_data = google_review_data_by_brand[brand_name]
        for url, price in zip(review_data['link'], review_data['price']):
            if id_initiator > 240:
                break
            google_scrape_review(brand_name, url, price, id_initiator)
            id_initiator = id_initiator + 1

#Function to get Yahoo Finance Stock Data
def get_stock_data_function():
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

    # Define the file path
    stock_csv_file_path = 'stock_data.csv'

    # Save the DataFrame to a CSV file with headers
    stock_data.to_csv(stock_csv_file_path, index=False)
# Function to load data

def load_data():
    
    # Read the amazon data files
    amazon_data_file_path = 'amazon_data123.csv'

    # Check if the file exists
    if not os.path.isfile(amazon_data_file_path):
        get_amazon_data_function()
    else:
        # Read the CSV file into a pandas DataFrame with headers
        amazon_data = pd.read_csv(amazon_data_file_path, names=['Brand', 'Shoe_ID', 'Price', 'Date', 'Ratings', 'Review'])
    
    # Adding '10' before every value in the 'Shoe_ID' column
    amazon_data['Shoe_ID'] = '10' + amazon_data['Shoe_ID'].astype(str)

    
    
    # Read the google data files
    google_data_file_path = 'google_data321.csv'

    # Check if the file exists
    if not os.path.isfile(google_data_file_path):
        get_google_data_function()
    else:
        # Read the CSV file into a pandas DataFrame with headers
        google_data = pd.read_csv(google_data_file_path, names=['Brand', 'Shoe_ID', 'Price', 'Date', 'Ratings', 'Review'])

    # Adding '10' before every value in the 'Shoe_ID' column
    google_data['Shoe_ID'] = '20' + google_data['Shoe_ID'].astype(str)

    
    
    # Concatenate Amazon and Google data
    amazon_google_data = pd.concat([amazon_data, google_data], ignore_index=True)

    # Read the dataset
    adidas_sales = pd.read_excel('Adidas Sales.xlsx', header=4)

    # Use only the specified columns
    adidas_sales = adidas_sales[['Retailer ID', 'Invoice Date' , 'State', 'City', 'Product', 'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit']]

    # Filter the data to include only rows where product is footwear
    adidas_sales = adidas_sales[adidas_sales['Product'].str.contains('Footwear', case=False)]
    
    # Rename the column 'Invoice Date' to 'Date'
    adidas_sales = adidas_sales.rename(columns={'Invoice Date': 'Date'})

    # Read the amazon data files
    stock_data_file_path = 'stock_data.csv'

    # Check if the file exists
    if not os.path.isfile(stock_data_file_path):
        get_stock_data_function()
    else:
        # Read the CSV file into a pandas DataFrame with headers
        stock_data = pd.read_csv(stock_data_file_path)

    # Merge stock data with review data
    combined_data = pd.merge(amazon_google_data, stock_data[['Date', 'Close', 'Brand']], on=['Date', 'Brand'], how='left')

    # Drop rows with missing values in the 'Close' column
    stock_analysis_data = combined_data.dropna(subset=['Close'])

    return stock_analysis_data, stock_data, adidas_sales

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
stock_analysis_data, stock_data, adidas_sales = load_data()

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Perform sentiment analysis
stock_analysis_data = perform_sentiment_analysis(stock_analysis_data)

# Sort data by date in ascending order
stock_analysis_data['Date'] = pd.to_datetime(stock_analysis_data['Date'])
stock_analysis_data = stock_analysis_data.sort_values(by='Date')

stock_analysis_data.dropna(inplace=True)

# Group the data by 'Invoice Date' and sum up 'Units Sold' and 'Total Sales'
adidas_sales_daywise = adidas_sales.groupby('Date')[['Units Sold', 'Total Sales']].sum().reset_index()

# Filter stock_data to create adidas_stock_data
adidas_stock_data = stock_data[stock_data['Brand'] == 'Adidas']

# Perform inner join on 'Date' column
adidas_merged_data = pd.merge(adidas_stock_data, adidas_sales_daywise, on='Date', how='inner')

# Select and rename columns
adidas_stock_analysis_data = adidas_merged_data[['Date', 'Units Sold', 'Total Sales', 'Close', 'Volume']]
adidas_stock_analysis_data = adidas_stock_analysis_data.rename(columns={'Units Sold_y': 'Units Sold', 'Total Sales_y': 'Total Sales', 'Volume': 'Stock Sold'})

# Function to get the brand logo
def get_brand_logo(selected_brand):
    # Dictionary mapping brands to their logos
    brand_logos = {
        'Adidas': 'Adidas_logo.png',
        'Nike': 'Nike_logo.png',
        'Skechers': 'Skechers_logo.png',
        'Under Armour': 'Under Armour_logo.png',
        'Puma': 'Puma_logo.png'
    }
    
    # Check if the selected brand exists in the dictionary
    if selected_brand in brand_logos:
        # Return the path to the logo image
        return brand_logos[selected_brand]


# Function to generate pie chart
def generate_pie_chart(data, selected_brand):
    sentiment_counts = data['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(sentiment_counts, autopct='%1.1f%%', startangle=90)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.title(f"Sentiment Analysis for {selected_brand}")
    plt.tight_layout()

    # Create legend without labels
    plt.legend(wedges, sentiment_counts.index, loc="upper right", labelspacing=0.5)
    
    return fig


# Function to generate line graph
def generate_line_graph(data, selected_brand, selected_year):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Close', data=data)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Stock Value for {selected_brand} in {selected_year}')
    plt.xticks(rotation=45)
    # Set x-axis interval to months
    plt.gca().xaxis.set_major_locator(MonthLocator())
    # Get the current figure and pass it to st.pyplot()   
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.pyplot(fig)
    #fig = plt.gcf()


def generate_dual_coloured_scatter(selected_brand_data, selected_brand):
    # Plotting
    plt.figure(figsize=(12, 6))

    # Define colors based on sentiment
    colors = selected_brand_data['Sentiment'].map({'Positive': 'blue', 'Negative': 'red', 'Neutral': 'gray'})

    # Scatter plot of Close price vs Date for Nike with colored points based on sentiment
    plt.scatter(selected_brand_data['Date'], selected_brand_data['Close'], c=colors, alpha=0.7)

    # Add legend
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Positive Sentiment'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Negative Sentiment'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Neutral Sentiment')
    ], loc='upper left')

    plt.title(f'{selected_brand} Stock Price over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


def generate_dual_line_graph_rescaled(data, selected_brand, selected_year):
    plt.figure(figsize=(10, 6))
    
    # Filter data for the selected brand and year
    selected_data = data[(data['Brand'] == selected_brand) & (data['Date'].dt.year == selected_year)]
    
    # Rescale ratings to match the range of close prices
    scaler = MinMaxScaler(feature_range=(selected_data['Close'].min(), selected_data['Close'].max()))
    selected_data['Scaled_Ratings'] = scaler.fit_transform(selected_data['Ratings'].values.reshape(-1, 1))
    
    # Plot rescaled ratings
    sns.lineplot(x='Date', y='Scaled_Ratings', data=selected_data, label='Scaled Ratings', color='green')
    
    # Plot close price
    sns.lineplot(x='Date', y='Close', data=selected_data, label='Close Price', color='red')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Rescaled Ratings vs Close Price for {selected_brand} in {selected_year}')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot()

def generate_adidas_double_line_graph(data):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    adidas_stock_analysis_data_2021 = data[adidas_stock_analysis_data['Date'].dt.year == 2021]

    # Scale and normalize the 'Total Sales' and 'Close' columns
    scaled_sales = scaler.fit_transform(adidas_stock_analysis_data_2021[['Units Sold']])
    scaled_close = scaler.fit_transform(adidas_stock_analysis_data_2021[['Close']])

    # Create a new DataFrame with the scaled values
    scaled_df = pd.DataFrame({'Date': adidas_stock_analysis_data_2021['Date'],
                            'Scaled Units Sold': scaled_sales.flatten(),
                            'Scaled Close': scaled_close.flatten()})

    # Plot the scaled total sales and close price
    plt.figure(figsize=(10, 6))
    plt.plot(scaled_df['Date'], scaled_df['Scaled Units Sold'], label='Scaled Units Sold', color='blue')
    plt.plot(scaled_df['Date'], scaled_df['Scaled Close'], label='Scaled Close Price', color='red')

    plt.xlabel('Date')
    plt.ylabel('Scaled Value')
    plt.title('Scaled Units Sold vs Scaled Close Price Over Time')
    plt.legend()
    st.pyplot(plt)

def generate_adidas_line_graph(line_adidas_sales_filtered, line_selected_state, line_selected_year):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Total Sales', data=line_adidas_sales_filtered)
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.title(f'Total Sales for {line_selected_state} in {line_selected_year}')
    plt.xticks(rotation=45)
    # Set x-axis interval to months
    plt.gca().xaxis.set_major_locator(MonthLocator())
    # Get the current figure and pass it to st.pyplot()   
    st.set_option('deprecation.showPyplotGlobalUse', False)

def create_wordcloud(data, selected_brand):
    
    # Define function to map POS tag to wordnet POS tag
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    # Assuming 'Selected_Brand' contains the selected brand name
    word_analysis_data = data[data['Brand'] == selected_brand].copy()

    # Tokenize, remove stopwords, and lemmatize the words in the 'Review' column
    words = []

    for review in word_analysis_data['Review']:
        tokens = word_tokenize(review)  # Tokenize the review
        words.extend(tokens)  # Extend the words list with tokens

    # Lowercase all words
    words = [w.lower() for w in words]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['shoe'])
    words = [w for w in words if w not in stop_words]

    # Remove punctuation
    words = [w for w in words if w.isalnum()]

    # Lemmatize words based on POS tags
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]

    # Create a single string of space separated words
    unique_string = " ".join(words)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400).generate(unique_string)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    #plt.show()
    st.pyplot(plt.gcf())

# Streamlit app
def main():
    st.title("Analyzing Shoe Brands' Sentiment, Sales & Stock Performance'")

    # Page navigation
    page = st.sidebar.radio("Navigate", ['Main Page', 'Explanation Page', 'Data Description', 'Adidas Analysis Page', 'Overall Analysis Page'])

    if page == 'Main Page':

        st.header('Taroon Ganesh - 7401376800')

        # 2. How to Use the WebApp:
        st.header("How to Use the WebApp:")

        st.markdown("""
            The features of this application:

            - **Navigation**: Use the sidebar on the left to navigate between different pages: "Main Page", "Explanation Page", "Adidas Analysis Page", and "Overall Analysis Page".
            
            - **Analysis Pages**: The "Explanation Page" contains details on what the app is and its specifications and findings. The "Adidas Analysis Page" focuses specifically on analyzing Adidas's data, including sales, sentiment analysis, and stock performance. The "Overall Analysis Page" allows you to select a specific brand and year to explore sentiment analysis, stock performance, and other insights.
            
            - **Interactivity**: You can select different brands and years to view specific analyses and insights. The webapp provides various interactive visualizations, including pie charts, line graphs, and word clouds, to help you understand consumer sentiment and stock performance.
            
            - **Conclusion**: After exploring the data and visualizations, you can draw conclusions about consumer sentiment, stock trends, and potential correlations between them. The webapp aims to provide actionable insights for improving products, marketing strategies, and brand perception based on the analysis results.
        """)

        # 3. Major "Gotchas":
        st.header("Major 'Gotchas':")

        st.markdown("""
            - **Limited Sales Data**: One major limitation of the project was the availability of sales data. We were only able to obtain sales data for the Adidas brand, and even for Adidas, the data was limited to the years 2020 and 2021. This limited our ability to perform comprehensive analysis and correlations across multiple brands and years.
            
            - **Misalignment of Review and Sales Data**: Another challenge was the misalignment of review and sales data. While we had review data spanning multiple years, the available sales data was limited to 2020 and 2021. This made it difficult to correlate sales trends with consumer sentiment accurately.
            
            - **Other Considerations**: Additionally, we encountered other minor challenges such as data preprocessing, handling missing values, and ensuring the accuracy of sentiment analysis results. However, through careful data processing and analysis, we were able to mitigate these challenges to some extent.
                    
            - **Used an env file to save my Google Shopping API key, but streamlit had problems installing that module. Hence, my API key is exposed in the code.
        """)

        

    elif page == 'Explanation Page':
        st.header('Explanation')
        
        st.header('What did you set out to study?')
        st.write("Initially, the project aimed to conduct sentiment analysis on consumer behavior using social media data from Twitter, product reviews from Amazon, and user-generated content from Reddit. However, the project evolved due:")

        st.write("- Originally, all three data sources primarily consisted of review data, limiting insights.")
        st.write("- Originally planned Twitter API usage was altered due to limitations and costs.")
        st.write("- Diverse data sources were chosen to provide richer insights beyond sentiment analysis.")

        st.write("**How it Differs from the Original Plan:**")
        st.write("- **Diverse Data Types:** Transitioned from solely analyzing review data to incorporating sales and stock data.")
        st.write("- **Change in Data Sources:** Shifted from relying solely on the Twitter API to using publicly available data from platforms like Amazon and Yahoo Finance.")
        st.write("- **Enhanced Analysis Scope:** Expanded analysis beyond sentiment to include factors influencing consumer behavior and brand performance.")

        st.header('What did you Discover/what were your conclusions?')

        st.markdown("#### Skechers Performance:")
        st.write("- **High Customer Ratings:** Despite its smaller market presence compared to Nike and Adidas, Skechers boasts the highest average customer ratings, around 4.54 stars.")
        st.write("- **Positive Reviews:** Skechers also enjoys the highest percentage of positive reviews, approximately 96%, which is quite impressive.")
        st.write("- **Stock Potential:** Despite its strong performance, Skechers maintains a relatively lower stock price, averaging around $50, suggesting a potentially lucrative investment opportunity.")

        st.markdown("#### Under Armour Investment Opportunity:")
        st.write("- **Low Risk:** Under Armour emerges as an attractive investment option with the lowest risk, given its cheapest stock price at $8.")
        st.write("- **Favorable Ratings:** Its average ratings, close to 4.4, indicate a favorable sentiment among customers.")

        st.markdown("#### Nike and Adidas:")
        st.write("- **Price Discrepancy:** Nike, despite having the costliest shoes with an average price of around $93, surprisingly exhibits the lowest percentage of positive reviews.")
        st.write("- **Brand Prestige:** Both Nike and Adidas establish themselves as prominent brands with the highest average stock prices compared to other brands.")

        st.markdown("#### Word Cloud Analysis:")
        st.write("- **Positive Attributes:** Analysis of word clouds for each brand reveals recurring positive attributes such as 'comfortable,' 'nice,' and 'great,' indicating strong consumer sentiment towards these brands.")

        st.markdown("#### Impact of Sentiment on Puma Stock Price:")
        st.write("- **Inconclusive Relationship:** The analysis explored the impact of sentiment on Puma's stock price, but conclusive evidence regarding the correlation between sentiment and stock prices could not be established.")

        st.markdown("#### Ratings vs Close Price Analysis:")
        st.write("- **Correlation Found:** Examining the relationship between ratings and close prices for each brand revealed a notable finding.")
        st.write("- **Correlation Trends:** Fluctuations in ratings tend to correlate with subsequent fluctuations in stock prices. Dips in ratings are often followed by dips in stock prices, while rises in ratings correspond to increases in stock prices in the subsequent months.")

        st.markdown("### Insights from Adidas Analysis:")

        st.markdown("#### Sales Analysis:")
        st.write("- **Total Sales and Profit:** Insights into the total sales, number of units sold, and overall profit of Adidas across the selected timeframe were obtained.")

        st.markdown("#### State-wise Performance:")
        st.write("- **Regional Variations:** Further analysis conducted on a state-wise basis revealed variations in performance.")
        st.write("- **Standout Performer:** New York emerged as a standout performer compared to other states, indicating stronger sales or market presence in that region.")

        st.markdown("#### Relationship between Units Sold and Close Price:")
        st.write("- **Significant Relationship:** A correlation analysis explored the relationship between units sold and the closing (stock) price of Adidas.")
        st.write("- **Visual Representation:** The correlation is visually depicted through a graph, showcasing how fluctuations in units sold correspond to changes in the close price of Adidas stock.")

        st.header('What difficulties did you have in completing the project?')
        st.write("Some difficulties encountered during the project included:")
        st.write("- Limited availability of sales data for multiple brands and recent timeframes, restricting the ability to perform comprehensive analysis across brands and time periods.")
        st.write("- Challenges in aligning review data with sales and stock data to derive meaningful correlations and insights.")
        st.write("- Initial reliance on paid APIs such as Twitter API, leading to a shift in data sources to obtain diverse data types.")

        st.header('What skills did you wish you had while you were doing the project?')
        st.write("I wish to have more advanced skills in data analysis and machine learning techniques. Especially, Machine Learning techniques that could have helped me with predictive analysis and more sophisticated data visualizations for deeper insights")

        st.header('What would you do “next” to expand or augment the project?')
        st.write("To expand the project further, the following steps could be taken:")
        st.write("- Obtain sales data for multiple brands in real-time or for recent timeframes to analyze the real-time performance of brands in the footwear industry.")
        st.write("- Incorporate additional data from social media platforms like Twitter and Reddit to capture public sentiment and discussions about brands, enabling a more comprehensive analysis of consumer behavior.")
        st.write("- Implement advanced analytics techniques such as predictive modeling and clustering to identify patterns, trends leading to more actionable insights for brands.")
    
    elif page == 'Data Description':
        st.header('Data Description')

        # Data Source 1
        st.write("**DATA SOURCE 1:**")
        st.markdown("- **API**: Twitter API")
        st.markdown("- **API Docs**: [Twitter API Documentation](https://docs.tweepy.org/en/stable/client.html)")
        st.write("**Brief Description:** The Twitter API allows access to a vast amount of social media data, including tweets, user profiles, follower counts, engagement metrics, and trends. This data can be used to analyze trends, sentiment, and the influence of social media on consumer behavior.")
        st.write("")

        # Data Source 2
        st.write("**DATA SOURCE 2:**")
        st.markdown("- **API:** Amazon Product Advertising API")
        st.markdown("- **API Docs:** [Amazon Product Advertising API Documentation](https://webservices.amazon.com/paapi5/documentation/)")
        st.write("**Brief Description:** The Amazon Product Advertising API provides access to product data, including product details, prices, ratings, reviews, and sales numbers. This data can be used to analyze consumer preferences, purchasing behavior, and the impact of social media influence on product sales.")
        st.write("")

        # Data Source 3
        st.write("**DATA SOURCE 3:**")
        st.markdown("- **Website:** Reddit")
        st.markdown("- **URL:** [Reddit Product Reviews](https://www.reddit.com/r/ProductReviews/)")
        st.write("**Brief Description:** Reddit is a social media platform with a wide range of subreddits covering diverse topics, including product reviews, consumer experiences, and discussions. By scraping data from relevant subreddits, we can collect user-generated content such as reviews, comments, and discussions related to various products and brands.")
        st.write("")

        # Data Source 4
        st.write("**Data Source 4:**")
        st.write("- **Dataset:** Adidas Sales Dataset")
        st.write("- **Dataset Link:** [Adidas Sales Dataset](https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset)")
        st.write("**Brief Description:** The Adidas Sales Dataset provides detailed information on Adidas sales, including the number of units sold, total sales revenue, sales locations, product types, and other relevant data. This dataset offers insights into Adidas's sales performance, market presence, and product preferences across different regions and product categories.")


    elif page == 'Adidas Analysis Page':
        st.header('Overall Analysis of Adidas')
        
        selected_brand = 'Adidas'

        st.write("")
        st.write("")

        # Display brand logo and calculate average price and ratings
        col1, col2 = st.columns(2)

        with col1:
            st.image(get_brand_logo(selected_brand), use_column_width='auto')

        with col2:

            available_years = [2020, 2021]
            selected_year = st.selectbox("Select Year", available_years)

            # Filter the adidas_sales DataFrame based on the selected year
            adidas_sales_selected_year = adidas_sales[adidas_sales['Date'].dt.year == selected_year]

            # Group the filtered DataFrame by the 'Date' column and aggregate the sales data
            sales_data_year = adidas_sales_selected_year.groupby('Date').agg({
                'Total Sales': 'sum',
                'Units Sold': 'sum',
                'Operating Profit': 'sum'
            }).reset_index()

            total_sales = sales_data_year['Total Sales'].sum()
            units_sold = sales_data_year['Units Sold'].sum()
            overall_operating_profit = sales_data_year['Operating Profit'].sum()

            # Function to convert numbers to K or M format
            def format_number(number):
                if number >= 1_000_000:
                    return f"${number / 1_000_000:.1f}M"
                elif number >= 1_000:
                    return f"${number / 1_000:.0f}K"
                else:
                    return f"${number:.0f}"

            # Convert total sales, units sold, and overall profit to K or M format
            total_sales_formatted = format_number(total_sales)
            units_sold_formatted = format_number(units_sold)
            overall_operating_profit_formatted = format_number(overall_operating_profit)

            # Display the stats
            st.markdown(f"<p style='text-align: center'><strong>Total Sales:</strong> {total_sales_formatted}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center'><strong>Units sold:</strong> {units_sold_formatted}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center'><strong>Overall Profit:</strong> {overall_operating_profit_formatted}</p>", unsafe_allow_html=True)


        st.write("")
        st.write("")

        st.subheader(f"Demographic Analysis: Statewise")

        # Display brand logo and calculate average price and ratings
        col1, col2 = st.columns(2)

        # Dropdown for selecting the state
        line_selected_state = col1.selectbox("Select State", ['All States'] + list(adidas_sales['State'].unique()))

        # Dropdown for selecting the year
        line_selected_year = col1.selectbox("Select Year", ['All Years'] + list(adidas_sales['Date'].dt.year.unique()))

        # Filter the adidas_sales DataFrame based on the selected state and year
        if line_selected_state != 'All States' and line_selected_year != 'All Years':
            line_adidas_sales_filtered = adidas_sales[(adidas_sales['State'] == line_selected_state) & (adidas_sales['Date'].dt.year == line_selected_year)]
        elif line_selected_state != 'All States':
            line_adidas_sales_filtered = adidas_sales[adidas_sales['State'] == line_selected_state]
        elif line_selected_year != 'All Years':
            line_adidas_sales_filtered = adidas_sales[adidas_sales['Date'].dt.year == line_selected_year]
        else:
            line_adidas_sales_filtered = adidas_sales

        # Group the filtered DataFrame by the 'Date' column and aggregate the sales data
        line_sales_data_month = line_adidas_sales_filtered.groupby(line_adidas_sales_filtered['Date'].dt.to_period('M')).agg({
            'Total Sales': 'sum'
        }).reset_index()

        # Calculate the total sales, units sold, and overall profit
        line_total_sales = line_sales_data_month['Total Sales'].sum()
        line_units_sold = line_adidas_sales_filtered['Units Sold'].sum()
        line_overall_profit = line_adidas_sales_filtered['Operating Profit'].sum()

        # Format total sales in terms of K for thousands and M for millions
        line_total_sales_formatted = f"${line_total_sales/1000:.1f}K" if line_total_sales < 1000000 else f"${line_total_sales/1000000:.1f}M"

        # Format units sold in terms of K for thousands
        line_units_sold_formatted = f"{line_units_sold/1000:.1f}K"

        # Format overall profit in terms of K for thousands and M for millions
        line_overall_profit_formatted = f"${line_overall_profit/1000:.1f}K" if line_overall_profit < 1000000 else f"${line_overall_profit/1000000:.1f}M"

        col2.write("")
        col2.write("")
        # Display the total sales, units sold, and overall profit
        col2.write(f"<p style='text-align: center'><strong>Total Sales:</strong> {line_total_sales_formatted}</p>", unsafe_allow_html=True)
        col2.write(f"<p style='text-align: center'><strong>Units sold:</strong> {line_units_sold_formatted}</p>", unsafe_allow_html=True)
        col2.write(f"<p style='text-align: center'><strong>Overall Profit:</strong> {line_overall_profit_formatted}</p>", unsafe_allow_html=True)

        # Display a line graph of total sales for every month for the selected state and year
        fig_lg = generate_adidas_line_graph(line_adidas_sales_filtered, line_selected_state, line_selected_year)
        st.pyplot(fig_lg)

        st.write("")
        st.write("")
        st.write("")

        st.subheader(f"Trend Analysis: Does Units Sold have an impact on Stock Price")
        st.write("")
        generate_adidas_double_line_graph(adidas_stock_analysis_data)


    elif page == 'Overall Analysis Page':
        st.header('Overall Analysis of all brands')

        # Dropdown for selecting brand
        selected_brand = st.selectbox("Select Brand", stock_analysis_data['Brand'].unique())

        # Filter data for selected brand
        selected_brand_data = stock_analysis_data[stock_analysis_data['Brand'] == selected_brand]

        # Dropdown for selecting year
        # available_years = selected_brand_data['Date'].dt.year.unique()
        available_years = [2023,2024]
        #selected_year = st.selectbox("Select Year", ['All Years'] + list(available_years))
        selected_year = st.selectbox("Select Year", available_years)

        # Filter data for selected year
        if selected_year != 'All Years':
            selected_brand_data = selected_brand_data[selected_brand_data['Date'].dt.year == selected_year]
            selected_stock_data = stock_data[(stock_data['Brand'] == selected_brand) & (stock_data['Date'].dt.year == selected_year)]
        else:
            selected_stock_data = stock_data[stock_data['Brand'] == selected_brand] 
        
        st.write("")
        st.write("")

        # Display brand logo and calculate average price and ratings
        col1, col2 = st.columns(2)

        with col1:
            st.image(get_brand_logo(selected_brand), use_column_width='auto')

        with col2:
            # Calculate average price and ratings
            avg_price = selected_brand_data['Price'].mean()
            avg_ratings = selected_brand_data['Ratings'].mean()

            # Calculate the number of blank lines needed for vertical centering
            num_blank_lines = st.empty()

            # Display blank lines for vertical centering
            for _ in range(4):
                st.write("")

            # Display average price and ratings
            st.markdown(f"<p style='text-align: center'><strong>Avg Price of Shoe:</strong> {avg_price:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center'><strong>Avg Ratings:</strong> {avg_ratings:.2f}</p>", unsafe_allow_html=True)

        st.write("")
        st.write("")

        # Generate pie chart
        st.subheader(f"Sentiment Analysis for {selected_brand}")
        st.write("")
        fig_pie = generate_pie_chart(selected_brand_data, selected_brand)
        st.pyplot(fig_pie)       

        st.write("")
        st.write("")

        # Generate line graph
        st.subheader(f"Stock Value for {selected_brand} in {selected_year}")
        st.write("")
        fig_lg = generate_line_graph(selected_stock_data, selected_brand, selected_year)
        st.pyplot(fig_lg)
        
        st.write("")
        st.write("")

        # Generate Wordcloud
        st.subheader(f"Most commen words for {selected_brand}")
        st.write("")
        create_wordcloud(stock_analysis_data, selected_brand)
        
        
        st.write("")
        st.write("")

        # Generate Dual Coloured Scatter plot
        st.subheader(f"Impact of Sensitivity on {selected_brand} Stock price")
        st.write("")
        generate_dual_coloured_scatter(selected_brand_data, selected_brand)

        st.write("")
        st.write("")

        # Generate dual line graph
        st.subheader(f"Ratings vs Close Price for {selected_brand} in {selected_year}")
        st.write("")
        generate_dual_line_graph_rescaled(selected_brand_data, selected_brand, selected_year)

if __name__ == "__main__":
    main()
