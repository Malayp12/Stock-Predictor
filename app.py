import datetime
import streamlit as st
import yfinance as yf
import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news
from bs4 import BeautifulSoup

# Cache the stock data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, start, end):
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                return None
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt+1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All download attempts failed: {e}")
                return None

# Cache news sentiment analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_news_sentiment(ticker):
    try:
        analyzer = SentimentIntensityAnalyzer()
        news_headlines = news.get_yf_rss(ticker)
        if not news_headlines:
            return 0.0
            
        news_sentiments = [analyzer.polarity_scores(article['title'])['compound'] for article in news_headlines[:10]]
        return sum(news_sentiments) / len(news_sentiments)
    except Exception as e:
        print(f"News sentiment error: {e}")
        return 0.0

# Cache Reddit sentiment analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_reddit_sentiment(ticker):
    try:
        reddit = praw.Reddit(
            client_id='FJutQldL2-xROrFQuskFbg',
            client_secret='yU07iUbu2sqAmJqtymhgtnv9jdv3ew',
            user_agent='myStockPredictorApp')

        subreddit = reddit.subreddit('stocks')
        analyzer = SentimentIntensityAnalyzer()
        reddit_sentiments = [analyzer.polarity_scores(post.title)['compound'] for post in subreddit.search(ticker, limit=10)]
        return sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else 0.0
    except Exception as e:
        print(f"Reddit sentiment error: {e}")
        return 0.0

# Cache Twitter sentiment analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_twitter_sentiment(ticker):
    try:
        response = requests.get(f"https://nitter.net/search?f=tweets&q=${ticker}&since=2024-01-01", 
                              headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        tweet_tags = soup.find_all('div', class_='tweet-content')
        tweets = [tag.text for tag in tweet_tags[:10]]
        
        if not tweets:
            return 0.0
            
        analyzer = SentimentIntensityAnalyzer()
        twitter_sentiments = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets]
        return sum(twitter_sentiments) / len(twitter_sentiments)
    except Exception as e:
        print(f"Twitter fetch error: {e}")
        return 0.0

# Cache the model training process
@st.cache_data(ttl=3600)  # Cache for 1 hour
def train_model(df, sentiment_score, window_size=60):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        price_window = scaled_data[i - window_size:i, 0]
        price_window = np.append(price_window, sentiment_score)
        X.append(price_window)
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Return everything needed for prediction and visualization
    return {
        'model': model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'split': split,
        'window_size': window_size,
        'scaled_data': scaled_data
    }

# Page setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (Linear Regression)")
st.write("Enter a stock ticker symbol to view historical stock prices and predict the next day's closing price.")

# User input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol:", value="TSLA").upper()
predict_button = st.sidebar.button("Predict")

# Display a status message during loading
if predict_button:
    with st.spinner(f"Analyzing {ticker} data and generating prediction..."):
        today = datetime.date.today()
        start_date = '2018-01-01'
        
        # Try to get the stock data with a descriptive error message
        df = get_stock_data(ticker, start=start_date, end=today)

        if df is None or df.empty:
            st.error(f"❌ No data found for {ticker}. Please check the stock symbol or try again later.")
        else:
            # Display historical chart
            st.markdown("---")
            st.subheader(f"{ticker} Historical Closing Price Chart")
            st.line_chart(df['Close'])
            
            # Get sentiment from different sources with error handling
            st.sidebar.markdown("### Sentiment Analysis")
            
            # News sentiment
            try:
                avg_news_sentiment = get_news_sentiment(ticker)
                st.sidebar.write(f"📰 Average News Sentiment: {avg_news_sentiment:.2f}")
            except Exception as e:
                st.sidebar.warning("⚠️ News sentiment analysis failed")
                avg_news_sentiment = 0.0
                
            # Reddit sentiment
            try:
                avg_reddit_sentiment = get_reddit_sentiment(ticker)
                st.sidebar.write(f"👾 Average Reddit Sentiment: {avg_reddit_sentiment:.2f}")
            except Exception as e:
                st.sidebar.warning("⚠️ Reddit sentiment analysis failed")
                avg_reddit_sentiment = 0.0
                
            # Twitter sentiment
            try:
                avg_twitter_sentiment = get_twitter_sentiment(ticker)
                st.sidebar.write(f"🐦 Average Twitter Sentiment: {avg_twitter_sentiment:.2f}")
            except Exception as e:
                st.sidebar.warning("⚠️ Twitter sentiment analysis failed")
                avg_twitter_sentiment = 0.0
            
            # Calculate combined sentiment
            available_sources = sum([
                1 if avg_news_sentiment != 0.0 else 0,
                1 if avg_reddit_sentiment != 0.0 else 0,
                1 if avg_twitter_sentiment != 0.0 else 0
            ])
            
            # Avoid division by zero
            if available_sources > 0:
                avg_combined_sentiment = (avg_news_sentiment + avg_reddit_sentiment + avg_twitter_sentiment) / available_sources
            else:
                avg_combined_sentiment = 0.0
                
            st.sidebar.write(f"🧠 Combined Sentiment Score: {avg_combined_sentiment:.2f}")
            
            # Train the model
            model_data = train_model(df, avg_combined_sentiment)
            
            # Extract model components
            model = model_data['model']
            scaler = model_data['scaler']
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            split = model_data['split']
            window_size = model_data['window_size']
            scaled_data = model_data['scaled_data']
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Rescale predictions and actual values
            y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Get dates for visualization
            dates = df.index[window_size + split:]
            
            # Plot predictions vs actual
            st.markdown("---")
            st.subheader(f"{ticker} Stock Price Prediction vs Actual")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, y_test_rescaled, label='Actual Price ($)')
            ax.plot(dates, y_pred_rescaled, label='Predicted Price ($)')
            ax.set_title(f'{ticker} Stock Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price (USD)')
            ax.legend()
            ax.grid()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Make tomorrow's prediction
            st.markdown("---")
            st.subheader(f"📅 Predicted Closing Price for Tomorrow ({ticker}):")
            
            # Get the last window_size days of data
            last_window_days = scaled_data[-window_size:]
            
            # Add sentiment score and reshape
            last_window_days_with_sentiment = np.append(last_window_days, avg_combined_sentiment).reshape(1, window_size + 1)
            
            # Make the prediction
            tomorrow_pred_scaled = model.predict(last_window_days_with_sentiment)
            
            # Rescale prediction
            tomorrow_pred_price = scaler.inverse_transform(tomorrow_pred_scaled.reshape(-1, 1))
            
            # Display prediction
            st.success(f"${tomorrow_pred_price[0][0]:.2f}")
            
            # Today's actual closing price for reference
            today_price = df['Close'].iloc[-1]
            st.info(f"Today's closing price: ${today_price:.2f}")
            
            # Calculate and display the predicted change
            predicted_change = tomorrow_pred_price[0][0] - today_price
            predicted_change_percent = (predicted_change / today_price) * 100
            
            if predicted_change > 0:
                st.success(f"Predicted change: +${predicted_change:.2f} (+{predicted_change_percent:.2f}%)")
            else:
                st.error(f"Predicted change: ${predicted_change:.2f} ({predicted_change_percent:.2f}%)")
            
            st.markdown("---")
            st.caption("Built by Malay Patel. Powered by Streamlit, Yahoo Finance, Reddit API.")
else:
    st.info("Enter a stock symbol and click 'Predict' to analyze the stock and generate a price prediction")

