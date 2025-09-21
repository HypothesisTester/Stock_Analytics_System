import pandas as pd

def _as_series(x):
    # Accept Series or single-column DataFrame (handles yfinance MultiIndex)
    if isinstance(x, pd.DataFrame):
        # take the first column if it’s a single field like Close/AAPL
        return x.iloc[:, 0]
    return pd.Series(x)

def SMA(series, timeperiod=20):
    s = _as_series(series)
    return s.rolling(window=timeperiod, min_periods=timeperiod).mean()

def EMA(series, timeperiod=20):
    s = _as_series(series)
    # TA-Lib uses EMA with alpha=2/(n+1); pandas ewm(span=n) matches this
    return s.ewm(span=timeperiod, adjust=False).mean()

def RSI(series, timeperiod=14):
    s = _as_series(series).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/timeperiod, adjust=False, min_periods=timeperiod).mean()
    avg_loss = loss.ewm(alpha=1/timeperiod, adjust=False, min_periods=timeperiod).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def MACD(series, fastperiod=12, slowperiod=26, signalperiod=9):
    s = _as_series(series)
    fast_ema = EMA(s, timeperiod=fastperiod)
    slow_ema = EMA(s, timeperiod=slowperiod)
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist
