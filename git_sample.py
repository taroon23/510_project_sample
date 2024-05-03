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

# Function to load data
def load_data():
    
    # Read the data files
    amazon_data = pd.read_csv('amazon_data123.csv', names=['Brand', 'Shoe_ID', 'Price', 'Date', 'Ratings', 'Review'])
    
    # Adding '10' before every value in the 'Shoe_ID' column
    amazon_data['Shoe_ID'] = '10' + amazon_data['Shoe_ID'].astype(str)

    google_data = pd.read_csv('google_data321.csv', names=['Brand', 'Shoe_ID', 'Price', 'Date', 'Ratings', 'Review'])

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

    stock_data = pd.read_csv('stock_data.csv')

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
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.title(f"Sentiment Analysis for {selected_brand}")
    plt.tight_layout()
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

# Function to generate Box Plot of Ratings by Brand
def generate_box_plot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Brand', y='Ratings', data=data)
    plt.xlabel('Brand')
    plt.ylabel('Ratings')
    plt.title('Box Plot of Ratings by Brand')
    plt.xticks(rotation=45)
    st.pyplot()

# Function to generate Bar Plot of Average Price by Brand
def generate_bar_plot(data):
    plt.figure(figsize=(10, 6))
    avg_price_by_brand = data.groupby('Brand')['Price'].mean().reset_index()
    sns.barplot(x='Brand', y='Price', data=avg_price_by_brand)
    plt.xlabel('Brand')
    plt.ylabel('Average Price')
    plt.title('Bar Plot of Average Price by Brand')
    plt.xticks(rotation=45)
    st.pyplot()

# Function to generate line graph of Ratings Over Time
def generate_line_plot_ratings(data, selected_brand, selected_year):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Ratings', data=data)
    plt.xlabel('Date')
    plt.ylabel('Ratings')
    plt.title(f'Ratings Over Time for {selected_brand} in {selected_year}')
    plt.xticks(rotation=45)
    # Set x-axis interval to months
    plt.gca().xaxis.set_major_locator(MonthLocator())
    st.pyplot()

# Function to generate Histogram of Close Price
def generate_close_price_histogram(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Close'], bins=20, kde=True)
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.title('Histogram of Close Price')
    st.pyplot()

# Function to generate Correlation Heatmap
def generate_correlation_heatmap(data):
    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=np.number)
    
    # Drop rows with missing values
    numeric_data.dropna(inplace=True)
    
    # Generate correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot()


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

# Streamlit app
def main():
    st.title('Stock Analysis Data App')

    # Page navigation
    page = st.sidebar.radio("Navigate", ['Main', 'Analysis Page', 'Adidas Analysis Page', 'Overall Analysis Page'])

    if page == 'Main':
        st.header('7401376800')

        # Overall Analysis Page
        st.subheader('Overall Analysis Page:')
        st.markdown("""
        - **Box Plot of Ratings by Brand:** Shows the distribution of ratings across different brands. It helps identify variability and central tendency in customer ratings.
        - **Bar Plot of Average Price by Brand:** Provides insight into the average price of shoes for each brand, allowing for easy comparison.
        - **Histogram of Close Price:** Visualizes the distribution of close prices across all shoes, giving an overview of price spread and frequency.
        - **Correlation Heatmap:** Illustrates correlations between numerical variables like close price, ratings, and price, aiding in identifying relationships.
        """)

        # Conclusions for Overall Analysis Page
        st.subheader('Conclusions:')
        st.write("""
        Through the overall analysis, users can understand the distribution of ratings, average prices, and close prices across brands, as well as explore correlations between different numerical variables. These insights help in identifying market trends, customer preferences, and potential areas for further investigation.
        """)

        # Explanation of the Brand Analysis Page
        st.subheader('Brand Analysis Page:')
        st.markdown("""
        - **Sentiment Analysis Pie Chart:** Represents the distribution of sentiment labels (positive, negative, neutral) based on customer reviews for the selected brand.
        - **Stock Value Line Graph:** Shows the trend of close prices for the selected brand over the chosen year, allowing for tracking of stock performance.
        - **Ratings Over Time Line Plot:** Illustrates changes in ratings for the selected brand over the selected year, helping identify customer satisfaction trends.
        """)

        # Conclusions for Brand Analysis Page
        st.subheader('Conclusions:')
        st.write("""
        The brand analysis page provides focused insights into a specific brand's customer sentiment, stock performance, and ratings trends over time. Users can use these insights to assess brand competitiveness, customer satisfaction, and market positioning.
        """)

    elif page == 'Analysis Page':
        st.header('Analysis Page')
        
        
    
    elif page == 'Adidas Analysis Page':
        st.header('Overall Analysis of Adidas')
        
        # Generate Box Plot of Ratings by Brand
        st.header('Box Plot of Ratings by Brand')
        generate_box_plot(stock_analysis_data)

        # Generate Bar Plot of Average Price by Brand
        st.header('Bar Plot of Average Price by Brand')
        generate_bar_plot(stock_analysis_data)

        # Generate Histogram of Close Price
        st.header('Histogram of Close Price')
        generate_close_price_histogram(stock_analysis_data)

        # Generate Correlation Heatmap
        st.header('Correlation Heatmap')
        generate_correlation_heatmap(stock_analysis_data)

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
        fig_pie = generate_pie_chart(selected_brand_data, selected_brand)
        st.pyplot(fig_pie)       

        # Generate line graph
        st.subheader(f"Stock Value for {selected_brand} in {selected_year}")
        fig_lg = generate_line_graph(selected_stock_data, selected_brand, selected_year)
        st.pyplot(fig_lg)
        
        # Generate line plot of Ratings Over Time
        st.subheader(f"Ratings Over Time for {selected_brand} in {selected_year}")
        generate_line_plot_ratings(selected_brand_data, selected_brand, selected_year)

        # Generate dual line graph
        st.subheader(f"Ratings vs Close Price for {selected_brand} in {selected_year}")
        generate_dual_line_graph_rescaled(selected_brand_data, selected_brand, selected_year)

if __name__ == "__main__":
    main()
