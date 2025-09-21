"""
Unit tests for the trading model module.

This test suite validates each component of the trading model module,
ensuring data collection, preprocessing, technical indicator
computation, news scraping, sentiment analysis, LSTM preparation and
confidence metric calculation all function correctly.  The tests
retain the original logic but reference the renamed symbols from
``my_trading_model.py``.  When TA‑Lib is unavailable, the tests
fall back to the local ``talib_compat`` shim provided in the project.
"""

# Use the standard library ``unittest`` module.  Remove any locally
# shadowing ``unittest`` from sys.modules so that imports resolve to
# the real standard library version.  Without this, a stray
# ``unittest.pyc`` in the project directory could be loaded and
# break ``unittest.mock`` imports.
import sys
import os
import glob

# Remove any compiled ``unittest`` bytecode in the local __pycache__ to
# prevent Python from loading a stale proxy.  Without this cleanup,
# ``import unittest`` could erroneously import a local bytecode file and
# break ``unittest.mock`` imports.
for _file in glob.glob(os.path.join(os.path.dirname(__file__), '__pycache__', 'unittest*.pyc')):
    try:
        os.remove(_file)
    except Exception:
        pass

# Remove any existing ``unittest`` entry from sys.modules to force a
# fresh import of the standard library module.
sys.modules.pop('unittest', None)

import unittest
from unittest.mock import patch

try:
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:
    # yfinance is optional; set to None so tests can be skipped gracefully
    yf = None  # type: ignore
import pandas as pd
import numpy as np

try:
    # Prefer the real TA‑Lib if installed.
    import talib  # type: ignore
except Exception:
    # Fall back to the local shim; behaves identically for the tested APIs.
    import talib_compat as talib  # type: ignore

# Import optional modules used in the main implementation.  They are
# not required for the unit tests themselves, but attempting to import
# them here can cause failures if the modules are not installed.
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
    from textblob import TextBlob  # type: ignore
except Exception:
    # If any of these optional dependencies are missing, the tests
    # themselves do not rely on them directly.  The functions under
    # test will handle missing dependencies as appropriate.
    pass
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

from my_trading_model import (
    fetch_yahoo_news,
    compute_sentiment_scores,
    prepare_dataset,
    compute_confidence_metric,
)


class TestTradingModel(unittest.TestCase):
    """Tests for the trading model's functions."""

    def setUp(self) -> None:
        # Set up mock data for use in various tests
        self.mock_stock_data = pd.DataFrame(
            {
                'Close': np.random.uniform(100, 200, 100),
                'Volume': np.random.randint(100000, 1000000, 100),
            }
        )

    @unittest.skipIf(yf is None, "yfinance not available")
    @patch('yfinance.download')
    def test_data_collection(self, mock_download: patch) -> None:
        """Ensure that yfinance.download returns a DataFrame and is not empty."""
        # Mock the download function to return the prepared DataFrame
        mock_download.return_value = self.mock_stock_data
        stock_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
        self.assertIsNotNone(stock_data)
        self.assertFalse(stock_data.empty)

    def test_data_preprocessing(self) -> None:
        """Verify that preprocessing fills missing data and adds return columns."""
        stock_data = pd.DataFrame({'Close': [1, 2, np.nan, 4, 5]})
        stock_data.ffill(inplace=True)
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        stock_data['Log_Return'] = np.log(
            stock_data['Close'] / stock_data['Close'].shift(1)
        )
        stock_data.dropna(inplace=True)
        self.assertEqual(stock_data['Close'].isnull().sum(), 0)
        self.assertIn('Daily_Return', stock_data.columns)
        self.assertIn('Log_Return', stock_data.columns)

    def test_technical_indicators(self) -> None:
        """Verify that technical indicator columns are added using talib or the shim."""
        stock_data = pd.DataFrame({'Close': np.random.uniform(100, 200, 100)})
        # SMA
        stock_data['SMA_20'] = talib.SMA(stock_data['Close'], timeperiod=20)
        # EMA
        stock_data['EMA_20'] = talib.EMA(stock_data['Close'], timeperiod=20)
        # RSI
        stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
        # MACD
        (
            stock_data['MACD'],
            stock_data['MACD_signal'],
            stock_data['MACD_hist'],
        ) = talib.MACD(stock_data['Close'])
        self.assertIn('SMA_20', stock_data.columns)
        self.assertIn('EMA_20', stock_data.columns)
        self.assertIn('RSI', stock_data.columns)
        self.assertIn('MACD', stock_data.columns)

    @patch('requests.get')
    def test_sentiment_analysis(self, mock_get: patch) -> None:
        """Ensure scraping and sentiment functions return non-empty results."""
        # Prepare a minimal HTML snippet to mock Yahoo news page
        mock_get.return_value.text = (
            '<html><body><article>'
            '<h3>Positive news for Apple</h3>'
            '<a href="https://example.com">link</a>'
            '</article></body></html>'
        )
        news_articles = fetch_yahoo_news('AAPL')
        sentiment_scores = compute_sentiment_scores(news_articles)
        self.assertGreater(len(news_articles), 0)
        self.assertGreater(len(sentiment_scores), 0)

    def test_rnn_preparation(self) -> None:
        """Check that prepare_dataset returns correctly shaped outputs and a scaler."""
        data = pd.DataFrame({'Close': np.random.uniform(100, 200, 100)})
        n_features = 60
        X_train, y_train, scaler = prepare_dataset(data, n_features)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.assertEqual(len(X_train), len(y_train))
        self.assertIsInstance(scaler, MinMaxScaler)

    def test_confidence_score_calculation(self) -> None:
        """Confirm that compute_confidence_metric returns a value between 0 and 1."""
        stock_data = pd.DataFrame(
            {
                'Close': [1, 2, 3, 4, 5],
                'RSI': [30, 40, 50, 60, 70],
                'MACD': [-0.5, 0, 0.5, 0, -0.5],
            }
        )
        sentiment_scores = [0.1, -0.2, 0.3]
        rnn_model = Sequential()
        # Simple dummy model to satisfy the interface; will not be trained
        rnn_model.add(Dense(1, input_dim=1))
        confidence_score = compute_confidence_metric(
            stock_data, sentiment_scores, rnn_model, MinMaxScaler(), 3
        )
        # The score should lie within a sensible range [0, 1]
        self.assertGreaterEqual(confidence_score, 0.0)
        self.assertLessEqual(confidence_score, 1.0)


if __name__ == '__main__':
    unittest.main()