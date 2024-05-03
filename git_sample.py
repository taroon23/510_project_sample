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

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet 
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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

        # Display brand logo and calculate average price and ratings
        col1, col2 = st.columns(2)

        with col1:
            st.write("")
            
            
        with col2:
           
            st.image(get_brand_logo(selected_brand), use_column_width='auto')
 

        st.write("")
        st.write("")

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

        # Display the stats
        st.markdown(f"<p style='text-align: center'><strong>Total Sales:</strong> {line_total_sales_formatted}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center'><strong>Units sold:</strong> {line_units_sold_formatted}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center'><strong>Overall Profit:</strong> {line_overall_profit_formatted}</p>", unsafe_allow_html=True)

        # Display a line graph of total sales for every month for the selected state and year
        st.subheader(f"Total Sales for {line_selected_state if line_selected_state != 'All States' else 'All States'} in {line_selected_year if line_selected_year != 'All Years' else 'All Years'}")
        st.line_chart(line_sales_data_month.set_index('Date'))


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

        # Generate line plot of Ratings Over Time
        st.subheader(f"Ratings Over Time for {selected_brand} in {selected_year}")
        st.write("")
        generate_line_plot_ratings(selected_brand_data, selected_brand, selected_year)

        st.write("")
        st.write("")

        # Generate dual line graph
        st.subheader(f"Ratings vs Close Price for {selected_brand} in {selected_year}")
        st.write("")
        generate_dual_line_graph_rescaled(selected_brand_data, selected_brand, selected_year)

if __name__ == "__main__":
    main()
