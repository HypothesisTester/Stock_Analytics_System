"""
Minorly adapted trading model implementation.

This module contains the core logic for fetching price data, computing
technical indicators, scraping news headlines, performing sentiment
analysis and training an LSTM neural network.  Names and comments have
been slightly adjusted from the original while preserving
functionality.  All heavy data fetching and model training is guarded
to run only when this module is executed as a script and when the
optional ``yfinance`` dependency is available.  When imported for
testing, only function definitions and compatibility wrappers are
evaluated, which avoids hard dependencies on ``yfinance`` and heavy
computation.
"""

from __future__ import annotations

# Attempt to import yfinance.  If unavailable, the variable ``yf``
# will be set to ``None`` and example code will be skipped at
# import time.  This allows unit tests to import this module without
# requiring yfinance to be installed.
try:
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:
    yf = None  # type: ignore

# Ticker symbol and date range used in the example workflow.  These
# constants are defined at module scope so that tests can reference
# them if needed.
ticker_sym: str = "AAPL"
start_dt: str = "2020-01-01"
end_dt: str = "2021-01-01"

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Try to import the real TA‑Lib library; fall back to the local shim if
# it is not installed.  This preserves functionality across systems
# where TA‑Lib may be missing.
try:
    import talib  # type: ignore
except Exception:
    import talib_compat as talib  # type: ignore


def fetch_yahoo_news(ticker: str) -> list[tuple[str, str]]:
    """
    Retrieve recent news articles from Yahoo Finance for the given ticker.

    The page is fetched via HTTP and parsed with BeautifulSoup to
    extract headline/link pairs.  The CSS selectors used here may
    require adjustment if Yahoo changes its markup.

    Args:
        ticker: Stock symbol (e.g. ``"AAPL"``).

    Returns:
        A list of tuples containing ``(headline, url)``.  If no articles
        are found the list will be empty.
    """
    url = f"https://news.yahoo.com/stock/{ticker}"
    response = requests.get(url)
    page_content = response.text
    soup = BeautifulSoup(page_content, 'html.parser')
    articles = soup.find_all('article')
    news_data: list[tuple[str, str]] = []
    for article in articles:
        headline_tag = article.find('h3')
        link_tag = article.find('a')
        if headline_tag and link_tag and link_tag.get('href'):
            news_data.append((headline_tag.get_text(), link_tag['href']))
    return news_data


def compute_sentiment_scores(news_articles: list[tuple[str, str]]) -> list[float]:
    """
    Compute sentiment polarity scores for a list of news headlines.

    Each headline is analyzed using TextBlob.  The polarity value
    returned ranges from ``-1`` (very negative) to ``1`` (very positive).

    Args:
        news_articles: List of ``(headline, url)`` tuples.

    Returns:
        A list of polarity scores corresponding to the headlines.  If no
        headlines are provided, a single neutral score of ``0.0`` is
        returned.
    """
    if not news_articles:
        return [0.0]
    scores: list[float] = []
    for headline, _ in news_articles:
        analysis = TextBlob(headline)
        scores.append(analysis.sentiment.polarity)
    return scores


def prepare_dataset(data: pd.DataFrame, n_features: int):
    """
    Prepare time-series data for input into an LSTM network.

    Selects the ``'Close'`` column, scales it to the ``[0, 1]`` range, and
    constructs sequences of length ``n_features`` to predict the next
    closing price.

    Args:
        data: DataFrame containing at least a ``'Close'`` column.
        n_features: Number of previous days to use for prediction.

    Returns:
        ``X_train``: 3D numpy array for LSTM input.
        ``y_train``: 1D numpy array of target values.
        ``scaler``: The fitted ``MinMaxScaler`` used for inverse transformations.
    """
    close_series = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_series)
    X_train: list = []
    y_train: list = []
    for i in range(n_features, len(scaled_data)):
        X_train.append(scaled_data[i - n_features:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train)
    X_train_arr = np.reshape(X_train_arr, (X_train_arr.shape[0], X_train_arr.shape[1], 1))
    return X_train_arr, y_train_arr, scaler


def compute_confidence_metric(
    stock_data: pd.DataFrame,
    sentiment_scores: list,
    rnn_model: Sequential,
    scaler: MinMaxScaler,
    n_features: int,
) -> float:
    """
    Calculate a composite confidence metric based on technical,
    sentiment and model predictions.  NaN‑safe.

    This function aggregates the latest RSI, MACD, news sentiment,
    and LSTM‑predicted price to derive a normalized confidence value.

    Args:
        stock_data: DataFrame containing technical indicators and closing prices.
        sentiment_scores: List of sentiment polarity values.
        rnn_model: Trained LSTM model for price prediction.
        scaler: MinMaxScaler instance used during training.
        n_features: Number of days used for prediction sequences.

    Returns:
        A float representing the confidence metric.
    """
    latest_data = stock_data.iloc[-n_features:]
    latest_close_prices = latest_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(latest_close_prices)
    X_test = np.array([scaled_data]).reshape(1, scaled_data.shape[0], 1)
    # Predict next closing price with fallback
    try:
        pred_price = rnn_model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
    except Exception:
        pred_price = latest_close_prices[-1:].astype(float)
    # Extract indicators
    latest_rsi = latest_data['RSI'].iloc[-1] if 'RSI' in latest_data.columns else np.nan
    latest_macd = latest_data['MACD'].iloc[-1] if 'MACD' in latest_data.columns else np.nan
    rsi_score = (latest_rsi / 100.0) if pd.notna(latest_rsi) else 0.5
    macd_raw = latest_macd if pd.notna(latest_macd) else 0.0
    macd_score = (macd_raw + 1.0) / 2.0
    avg_sent = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
    last_close = float(latest_close_prices[-1][0])
    rel_move = float(pred_price[0][0] / last_close) if last_close else 1.0
    parts = np.array([rsi_score, macd_score, avg_sent, rel_move], dtype=float)
    parts = np.nan_to_num(parts, nan=0.5)
    return float(parts.mean())


# -----------------------------------------------------------------------------
# Backwards compatibility wrappers.  These alias names mirror those in the
# original project so that existing imports continue to work.  They simply
# delegate to the updated function names defined above.

def scrape_yahoo_news(*args, **kwargs):
    """Alias for ``fetch_yahoo_news`` for backwards compatibility."""
    return fetch_yahoo_news(*args, **kwargs)


def analyze_sentiment(*args, **kwargs):
    """Alias for ``compute_sentiment_scores`` for backwards compatibility."""
    return compute_sentiment_scores(*args, **kwargs)


def prepare_data(*args, **kwargs):
    """Alias for ``prepare_dataset`` for backwards compatibility."""
    return prepare_dataset(*args, **kwargs)


def calculate_confidence_score(*args, **kwargs):
    """Alias for ``compute_confidence_metric`` for backwards compatibility."""
    return compute_confidence_metric(*args, **kwargs)


# -----------------------------------------------------------------------------
# Example workflow: only runs when this module is executed directly and
# when ``yfinance`` is available.  This prevents heavy data downloads
# and model training from occurring when the module is imported for testing.

if __name__ == '__main__' and yf is not None:
    # Fetch historical stock data
    stock_info = yf.download(ticker_sym, start=start_dt, end=end_dt)
    print(stock_info.head())

    # Check for missing values
    missing_values = stock_info.isnull().sum()
    print("Missing values in each column:\n", missing_values)

    # Preprocess: forward-fill and compute returns
    stock_info.ffill(inplace=True)
    stock_info['Daily_Return'] = stock_info['Close'].pct_change()
    stock_info['Log_Return'] = np.log(stock_info['Close'] / stock_info['Close'].shift(1))
    stock_info.dropna(inplace=True)
    print(stock_info.head())

    # Calculate technical indicators
    stock_info['SMA_20'] = talib.SMA(stock_info['Close'], timeperiod=20)
    stock_info['SMA_50'] = talib.SMA(stock_info['Close'], timeperiod=50)
    stock_info['EMA_20'] = talib.EMA(stock_info['Close'], timeperiod=20)
    stock_info['EMA_50'] = talib.EMA(stock_info['Close'], timeperiod=50)
    stock_info['RSI'] = talib.RSI(stock_info['Close'], timeperiod=14)
    stock_info['MACD'], stock_info['MACD_signal'], stock_info['MACD_hist'] = talib.MACD(
        stock_info['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    print(
        stock_info[
            [
                'Close',
                'SMA_20',
                'SMA_50',
                'EMA_20',
                'EMA_50',
                'RSI',
                'MACD',
                'MACD_signal',
                'MACD_hist',
            ]
        ].head()
    )

    # Fetch news and compute sentiment
    news_articles = fetch_yahoo_news(ticker_sym)
    sentiment_scores = compute_sentiment_scores(news_articles)

    # Prepare dataset and train model
    n_features = 60
    X_train, y_train, scaler = prepare_dataset(stock_info, n_features)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Compute and print confidence score
    score = compute_confidence_metric(stock_info, sentiment_scores, model, scaler, n_features)
    print("Confidence Score for the stock:", score)
