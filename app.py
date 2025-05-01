import datetime
import streamlit as st
import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news
from bs4 import BeautifulSoup

# Alpha Vantage API Setup
API_KEY = "NTIUPIA31V3JCJOG"

def get_alpha_vantage_data(symbol, api_key):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        st.error(f"Alpha Vantage response error: {data}")
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df

# Page setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (Linear Regression)")
st.write("Enter a stock ticker symbol to view historical stock prices and predict the next day's closing price.")

# User input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol:", value="TSLA").upper()
predict_button = st.sidebar.button("Predict")

if predict_button:
    today = datetime.date.today()
    df = get_alpha_vantage_data(ticker, API_KEY)

    if df is None:
        st.error("❌ Failed to load stock data from Alpha Vantage.")
    else:
        df = df[df.index <= pd.to_datetime(today)]
        st.markdown("---")
        st.subheader(f"{ticker} Historical Closing Price Chart")
        st.line_chart(df['Close'])

        analyzer = SentimentIntensityAnalyzer()
        news_headlines = news.get_yf_rss(ticker)
        avg_news_sentiment = 0.0

        if news_headlines:
            news_sentiments = [analyzer.polarity_scores(article['title'])['compound'] for article in news_headlines[:10]]
            avg_news_sentiment = sum(news_sentiments) / len(news_sentiments)

        st.sidebar.write(f"📰 Average News Sentiment Score: {avg_news_sentiment:.2f}")

        reddit = praw.Reddit(
            client_id='FJutQldL2-xROrFQuskFbg',
            client_secret='yU07iUbu2sqAmJqtymhgtnv9jdv3ew',
            user_agent='myStockPredictorApp')

        subreddit = reddit.subreddit('stocks')
        reddit_sentiments = [analyzer.polarity_scores(post.title)['compound'] for post in subreddit.search(ticker, limit=10)]
        avg_reddit_sentiment = sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else 0.0
        st.sidebar.write(f"👾 Average Reddit Sentiment Score: {avg_reddit_sentiment:.2f}")

        tweets = []
        try:
            response = requests.get(f"https://nitter.net/search?f=tweets&q=${ticker}&since=2024-01-01")
            soup = BeautifulSoup(response.text, 'html.parser')
            tweet_tags = soup.find_all('div', class_='tweet-content')
            tweets = [tag.text for tag in tweet_tags[:10]]
        except Exception as e:
            print(f"Twitter fetch error: {e}")

        twitter_sentiments = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets]
        avg_twitter_sentiment = sum(twitter_sentiments) / len(twitter_sentiments) if twitter_sentiments else 0.0
        st.sidebar.write(f"🐦 Average Twitter Sentiment Score: {avg_twitter_sentiment:.2f}")

        avg_combined_sentiment = (avg_news_sentiment + avg_reddit_sentiment + avg_twitter_sentiment) / 3
        st.sidebar.write(f"🧠 Combined Sentiment Score: {avg_combined_sentiment:.2f}")

        data = df[['Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        window_size = 60
        for i in range(window_size, len(scaled_data)):
            price_window = scaled_data[i - window_size:i, 0]
            price_window = np.append(price_window, avg_combined_sentiment)
            X.append(price_window)
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        dates = df.index[window_size + split:]

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

        st.markdown("---")
        st.subheader(f"📅 Predicted Closing Price for Tomorrow ({ticker}):")
        last_60_days = scaled_data[-window_size:]
        last_60_days = np.append(last_60_days, avg_combined_sentiment).reshape(1, window_size + 1)
        tomorrow_pred_scaled = model.predict(last_60_days)
        tomorrow_pred_price = scaler.inverse_transform(tomorrow_pred_scaled.reshape(-1, 1))
        st.success(f"${tomorrow_pred_price[0][0]:.2f}")

        st.markdown("---")
        st.caption("Built by Malay Patel. Powered by Streamlit, Alpha Vantage, Reddit API.")