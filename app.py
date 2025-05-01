import datetime
import streamlit as st
import yfinance as yf
import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news
from bs4 import BeautifulSoup

# =====================================================================
# CREATE PERSISTENT SESSION TO REDUCE RATE LIMITING
# =====================================================================
# Create a persistent session for yfinance to reduce rate limiting impact
@st.cache_resource
def get_yf_session():
    session = requests.Session()
    return session

# Initialize persistent session
yf_session = get_yf_session()
yf.set_tz_session_api(session=yf_session)

# =====================================================================
# ENHANCED DATA FETCHING WITH FALLBACK MECHANISMS
# =====================================================================
# Cache the stock data with multiple fallback mechanisms
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, start, end):
    # Multiple attempts with different methods
    methods = [
        {"method": "yfinance_direct", "retry_count": 3, "initial_delay": 2},
        {"method": "yahoo_fin_fallback", "retry_count": 2, "initial_delay": 3}
    ]
    
    last_error = None
    
    # Try each method until one succeeds
    for method_info in methods:
        method = method_info["method"]
        retry_count = method_info["retry_count"]
        delay = method_info["initial_delay"]
        
        # Multiple retries for each method
        for attempt in range(retry_count):
            try:
                if method == "yfinance_direct":
                    # First approach: direct yfinance download
                    df = yf.download(
                        ticker, 
                        start=start, 
                        end=end, 
                        progress=False,
                        session=yf_session
                    )
                    
                    if not df.empty:
                        return df
                    
                elif method == "yahoo_fin_fallback":
                    # Second approach: use period instead of start/end dates
                    df = yf.download(
                        ticker, 
                        period="5y",  # Use period instead of explicit dates
                        progress=False,
                        session=yf_session
                    )
                    
                    if not df.empty:
                        # Filter to desired date range after download
                        df = df[start:end]
                        if not df.empty:
                            return df
                
                # If we get here, the current attempt failed
                print(f"{method} attempt {attempt+1} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                
            except Exception as e:
                last_error = e
                print(f"{method} attempt {attempt+1} error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    # If we get here, all methods failed
    print(f"All download attempts failed. Last error: {last_error}")
    
    # Final fallback: try to get just recent data (past week)
    try:
        df = yf.download(ticker, period="1wk", progress=False, session=yf_session)
        if not df.empty:
            st.warning("⚠️ Limited to recent data only due to API limitations.")
            return df
    except Exception as e:
        print(f"Final fallback attempt failed: {e}")
    
    return None

# Cache news sentiment analysis with error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_news_sentiment(ticker):
    try:
        analyzer = SentimentIntensityAnalyzer()
        news_headlines = news.get_yf_rss(ticker)
        
        if not news_headlines:
            print("No news headlines found")
            return 0.0
            
        news_sentiments = [analyzer.polarity_scores(article['title'])['compound'] for article in news_headlines[:10]]
        return sum(news_sentiments) / len(news_sentiments)
    except Exception as e:
        print(f"News sentiment error: {e}")
        return 0.0

# Cache Reddit sentiment analysis with error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_reddit_sentiment(ticker):
    try:
        reddit = praw.Reddit(
            client_id='FJutQldL2-xROrFQuskFbg',
            client_secret='yU07iUbu2sqAmJqtymhgtnv9jdv3ew',
            user_agent='myStockPredictorApp')

        subreddit = reddit.subreddit('stocks')
        analyzer = SentimentIntensityAnalyzer()
        
        # Add randomized sleep to avoid rate limiting
        time.sleep(random.uniform(1.0, 2.0))
        
        reddit_sentiments = [analyzer.polarity_scores(post.title)['compound'] for post in subreddit.search(ticker, limit=10)]
        return sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else 0.0
    except Exception as e:
        print(f"Reddit sentiment error: {e}")
        return 0.0

# Cache Twitter sentiment analysis with error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_twitter_sentiment(ticker):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'DNT': '1',  # Do Not Track
        }
        
        # Add randomized sleep to avoid rate limiting
        time.sleep(random.uniform(1.0, 2.0))
        
        response = requests.get(
            f"https://nitter.net/search?f=tweets&q=${ticker}&since=2024-01-01", 
            headers=headers
        )
        
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
    # Handle case when df is too small
    if len(df) < window_size + 10:
        window_size = max(5, len(df) // 3)  # Use at least 5 days, or 1/3 of available data
        st.warning(f"⚠️ Limited data available. Using {window_size} days for prediction window.")
    
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
    
    # Handle case when X is too small for an 80/20 split
    if len(X) < 10:
        split = max(1, int(len(X) * 0.5))  # Use at least 1 item for testing
    else:
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

# =====================================================================
# PAGE SETUP AND USER INTERFACE
# =====================================================================
# Page setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (Linear Regression)")
st.write("Enter a stock ticker symbol to view historical stock prices and predict the next day's closing price.")

# User input
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol:", value="TSLA").upper()
period_options = {
    "Recent Data Only (1 week)": "1wk",
    "1 Month": "1mo", 
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years (may be rate limited)": "5y"
}
selected_period = st.sidebar.selectbox(
    "Data Time Period:", 
    options=list(period_options.keys()),
    index=3  # Default to 6 months
)
period = period_options[selected_period]

predict_button = st.sidebar.button("Predict")

# Display a status message during loading
if predict_button:
    with st.spinner(f"Analyzing {ticker} data and generating prediction..."):
        today = datetime.date.today()
        
        # Calculate start date based on period
        if period == "1wk":
            start_date = today - datetime.timedelta(days=7)
        elif period == "1mo":
            start_date = today - datetime.timedelta(days=30)
        elif period == "3mo":
            start_date = today - datetime.timedelta(days=90)
        elif period == "6mo":
            start_date = today - datetime.timedelta(days=180)
        elif period == "1y":
            start_date = today - datetime.timedelta(days=365)
        elif period == "2y":
            start_date = today - datetime.timedelta(days=730)
        else:  # 5y
            start_date = today - datetime.timedelta(days=1825)
        
        # Try to get the stock data with a descriptive error message
        try:
            if period == "1wk" or period == "1mo" or period == "3mo":
                # For shorter periods, use period parameter directly to reduce API calls
                df = yf.download(ticker, period=period, progress=False, session=yf_session)
            else:
                # For longer periods, try the more robust function
                df = get_stock_data(ticker, start=start_date, end=today)
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            df = None

        if df is None or df.empty:
            st.error(f"❌ No data found for {ticker}. Please check the stock symbol or try again later.")
            st.info("💡 Tip: Try a shorter time period or a different stock symbol.")
        else:
            # Check data quality
            if len(df) < 5:
                st.warning(f"⚠️ Very limited data available for {ticker} ({len(df)} days). Predictions may be unreliable.")
            
            # Display historical chart
            st.markdown("---")
            st.subheader(f"{ticker} Historical Closing Price Chart")
            st.line_chart(df['Close'])
            
            # Get sentiment from different sources with error handling
            st.sidebar.markdown("### Sentiment Analysis")
            
            # News sentiment
            with st.sidebar:
                with st.spinner("Analyzing news sentiment..."):
                    try:
                        avg_news_sentiment = get_news_sentiment(ticker)
                        st.write(f"📰 Average News Sentiment: {avg_news_sentiment:.2f}")
                    except Exception as e:
                        st.warning("⚠️ News sentiment analysis failed")
                        avg_news_sentiment = 0.0
                    
            # Reddit sentiment
            with st.sidebar:
                with st.spinner("Analyzing Reddit sentiment..."):
                    try:
                        avg_reddit_sentiment = get_reddit_sentiment(ticker)
                        st.write(f"👾 Average Reddit Sentiment: {avg_reddit_sentiment:.2f}")
                    except Exception as e:
                        st.warning("⚠️ Reddit sentiment analysis failed")
                        avg_reddit_sentiment = 0.0
                    
            # Twitter sentiment
            with st.sidebar:
                with st.spinner("Analyzing Twitter sentiment..."):
                    try:
                        avg_twitter_sentiment = get_twitter_sentiment(ticker)
                        st.write(f"🐦 Average Twitter Sentiment: {avg_twitter_sentiment:.2f}")
                    except Exception as e:
                        st.warning("⚠️ Twitter sentiment analysis failed")
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
                st.sidebar.warning("⚠️ No sentiment data available. Using neutral sentiment (0.0)")
                
            st.sidebar.write(f"🧠 Combined Sentiment Score: {avg_combined_sentiment:.2f}")
            
            # Train the model with adaptive window size
            try:
                model_data = train_model(df, avg_combined_sentiment)
                
                # Extract model components
                model = model_data['model']
                scaler = model_data['scaler']
                X_test = model_data['X_test']
                y_test = model_data['y_test']
                split = model_data['split']
                window_size = model_data['window_size']
                scaled_data = model_data['scaled_data']
                
                # Make predictions if there's enough test data
                if len(X_test) > 0 and len(y_test) > 0:
                    y_pred = model.predict(X_test)
                    
                    # Rescale predictions and actual values
                    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
                    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Get dates for visualization
                    dates = df.index[window_size + split:]
                    
                    # Plot predictions vs actual
                    st.markdown("---")
                    st.subheader(f"{ticker} Stock Price Prediction vs Actual")
                    
                    # Create the plot
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
                else:
                    st.warning("⚠️ Not enough data for test vs prediction visualization")
                    
                # Make tomorrow's prediction
                st.markdown("---")
                st.subheader(f"📅 Predicted Closing Price for Tomorrow ({ticker}):")
                
                # Get the last window_size days of data
                if len(scaled_data) >= window_size:
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
                    
                    # Model accuracy information
                    if len(X_test) > 0 and len(y_test) > 0:
                        # Calculate and display model accuracy metrics if we have test data
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                        r2 = r2_score(y_test_rescaled, y_pred_rescaled)
                        
                        st.markdown("---")
                        st.subheader("Model Performance Metrics:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
                        with col2:
                            st.metric("Root Mean Squared Error", f"${rmse:.2f}")
                        with col3:
                            st.metric("R² Score", f"{r2:.3f}")
                            
                        # Interpret the accuracy
                        if r2 < 0.3:
                            st.warning("⚠️ Model has low predictive power. Consider using more data.")
                        elif r2 >= 0.7:
                            st.success("✅ Model shows good predictive performance.")
                else:
                    st.error(f"⚠️ Not enough historical data ({len(scaled_data)} days) for prediction with window size {window_size}")
                    
            except Exception as e:
                st.error(f"❌ Error in prediction model: {e}")
                st.info("💡 Try using a longer time period or a different stock symbol.")
            
            st.markdown("---")
            st.caption("Built by Malay Patel. Powered by Streamlit, Yahoo Finance, Reddit API.")
else:
    st.info("Enter a stock symbol and click 'Predict' to analyze the stock and generate a price prediction")
    
    # Show a demo placeholder chart
    st.markdown("### 📊 Sample Visualization")
    demo_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'price': np.random.normal(100, 10, 100).cumsum() + 500
    }).set_index('date')
    
    st.line_chart(demo_data)
    
    st.markdown("""
    ### How It Works
    This app predicts tomorrow's stock closing price using:
    - Historical stock price data from Yahoo Finance 📊
    - Sentiment analysis from News 📰, Reddit 👾, and Twitter 🐦
    - Linear Regression machine learning model 🧠
    
    Enter a stock symbol above and click "Predict" to get started!
    """)
