import datetime
import streamlit as st
import yfinance as yf
import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news
from bs4 import BeautifulSoup

# Page setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("\U0001F4C8 Stock Price Predictor (Linear Regression)")
st.write("Enter a stock ticker symbol to view historical stock prices and predict the next day's closing price.")

# User input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol:", value="TSLA")
if not ticker:
    ticker = "TSLA"
predict_button = st.sidebar.button("Predict")

# Button
if predict_button:
    today = datetime.date.today()

    try:
        st.write(f"DEBUG - Ticker entered: {ticker}")
        df = yf.download(ticker, start='2018-01-01', end=today, progress=False)

        if df.empty:
            st.error("\u274C No data found. Please check the stock symbol or try again later.")
        else:
            # === Show Historical Chart ===
            st.markdown("---")
            st.subheader(f"{ticker} Historical Closing Price Chart")
            st.line_chart(df['Close'])

            # === Fetch News and Calculate News Sentiment ===
            analyzer = SentimentIntensityAnalyzer()
            news_headlines = news.get_yf_rss(ticker)

            if not news_headlines:
                avg_news_sentiment = 0.0
            else:
                news_sentiments = []
                for article in news_headlines[:10]:
                    title = article['title']
                    sentiment = analyzer.polarity_scores(title)
                    news_sentiments.append(sentiment['compound'])
                avg_news_sentiment = sum(news_sentiments) / len(news_sentiments)

            st.sidebar.write(f"\U0001F4F0 Average News Sentiment Score: {avg_news_sentiment:.2f}")

            # === Fetch Reddit Posts and Calculate Reddit Sentiment ===
            reddit = praw.Reddit(
                client_id='FJutQldL2-xROrFQuskFbg',
                client_secret='yU07iUbu2sqAmJqtymhgtnv9jdv3ew',
                user_agent='myStockPredictorApp'
            )

            subreddit = reddit.subreddit('stocks')
            posts = subreddit.search(ticker, limit=10)

            reddit_sentiments = []
            for post in posts:
                sentiment = analyzer.polarity_scores(post.title)
                reddit_sentiments.append(sentiment['compound'])

            avg_reddit_sentiment = sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else 0.0
            st.sidebar.write(f"\U0001F47E Average Reddit Sentiment Score: {avg_reddit_sentiment:.2f}")

            # === Fetch Tweets (web scraped from nitter.net) ===
            tweets = []
            search_url = f"https://nitter.net/search?f=tweets&q=${ticker}&since=2024-01-01"

            try:
                response = requests.get(search_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                tweet_tags = soup.find_all('div', class_='tweet-content')

                for tag in tweet_tags[:10]:
                    tweets.append(tag.text)

            except Exception as e:
                print(f"Error fetching tweets: {e}")

            twitter_sentiments = []
            for tweet_text in tweets:
                sentiment = analyzer.polarity_scores(tweet_text)
                twitter_sentiments.append(sentiment['compound'])

            avg_twitter_sentiment = sum(twitter_sentiments) / len(twitter_sentiments) if twitter_sentiments else 0.0
            st.sidebar.write(f"\U0001F426 Average Twitter Sentiment Score: {avg_twitter_sentiment:.2f}")

            # === Combine Sentiments ===
            avg_combined_sentiment = (avg_news_sentiment + avg_reddit_sentiment + avg_twitter_sentiment) / 3
            st.sidebar.write(f"\U0001F9E0 Combined Sentiment Score: {avg_combined_sentiment:.2f}")

            # === Prepare data for prediction ===
            data = df[['Close']].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            X = []
            y = []
            window_size = 60

            for i in range(window_size, len(scaled_data)):
                price_window = scaled_data[i - window_size:i, 0]
                price_window = np.append(price_window, avg_combined_sentiment)
                X.append(price_window)
                y.append(scaled_data[i, 0])

            X = np.array(X)
            y = np.array(y)

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

            # === Predict Tomorrow's Price ===
            st.markdown("---")
            st.subheader(f"\U0001F4C5 Predicted Closing Price for Tomorrow ({ticker}):")

            last_60_days = scaled_data[-window_size:]
            last_60_days = np.append(last_60_days, avg_combined_sentiment)
            last_60_days = last_60_days.reshape(1, window_size + 1)

            tomorrow_pred_scaled = model.predict(last_60_days)
            tomorrow_pred_price = scaler.inverse_transform(tomorrow_pred_scaled.reshape(-1, 1))

            st.success(f"${tomorrow_pred_price[0][0]:.2f}")

            st.markdown("---")
            st.caption("Built by Malay Patel. Powered by Streamlit, Yahoo Finance, Reddit API.")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for {ticker}: {e}")

