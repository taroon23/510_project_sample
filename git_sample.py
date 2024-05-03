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
