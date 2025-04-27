# 📈 Stock Price Predictor with Multi-Source Sentiment Analysis

A powerful machine learning application that predicts tomorrow’s stock closing price using:
- Past 60 days of stock price history 📊
- Live Sentiment Analysis from News 📰, Reddit 👾, and Twitter 🐦

This project uses **Linear Regression** combined with **multi-source market sentiment** to make smarter stock predictions — simulating real-world trading strategies used by hedge funds and analysts.

---

## 🌍 Live Demo

🚀 [Click here to try the app!](https://stock-predictor-jbrkpzwuwtwbqcrxgpsiu5.streamlit.app/)

---

## ✨ Features

- 📈 View historical closing prices for any stock symbol
- 📰 Analyze market sentiment from Yahoo News headlines
- 👾 Analyze sentiment from Reddit posts
- 🐦 Analyze sentiment from Twitter posts
- 🧠 Combine all sentiments into a final score
- 🔮 Predict tomorrow's stock closing price based on price trends + today's market mood
- 🎨 Beautiful Streamlit interface — mobile friendly and responsive

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:
  - `yfinance` for stock data
  - `scikit-learn` for Linear Regression modeling
  - `vaderSentiment` for sentiment analysis
  - `yahoo_fin` for pulling news articles
  - `praw` for Reddit API
  - `BeautifulSoup` and `requests` for Twitter scraping
  - `matplotlib` for visualization
- **Deployment**: Streamlit Cloud

---

## 📋 Future Improvements (Version 2 Ideas)

- Upgrade model to LSTM (Recurrent Neural Networks) for sequence prediction
- Add real-time stock price updates
- Add live news sentiment updates every few minutes
- Train separate models for high-sentiment vs low-sentiment situations

---

## 📚 Development Process

This project was developed following a two-stage professional workflow:

- [Prototype Development Notebook](./prototype/prototype.ipynb):  
  Early experimentation with stock price prediction using Linear Regression on Tesla data (TSLA).
- Final Deployment:  
  The Streamlit app (`app.py`) connects live stock prices with multi-source sentiment analysis to predict future closing prices.

## 🧑‍💻 About the Developer

Built by **Malay Patel**  
- 2nd Year Computer Science Student @ Wilfrid Laurier University
- Passionate about Machine Learning, Finance, and AI-driven solutions!

🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/malaypatel12)

---

> **"Predict smarter. Trade smarter. Think smarter."** 🚀
