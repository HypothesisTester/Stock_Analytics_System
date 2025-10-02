# Stock Analytics System

## Overview
A compact end-to-end workflow for exploring equity data. The project stitches together technical indicators, headline sentiment, and an LSTM forecaster to produce a blended confidence signal for a single ticker. The default example targets Apple stock, but every component is written to accept arbitrary tickers and data frames.

## Capabilities
- **Market data ingestion:** Pulls daily price history with `yfinance` and guards the import so unit tests can run without the package.
- **Feature engineering:** Computes SMA/EMA, RSI, MACD and basic return series, falling back to `talib_compat.py` when the native TA-Lib wheel is unavailable.
- **News + sentiment:** Scrapes Yahoo Finance headlines, scores them with TextBlob, and tolerates empty or unavailable feeds.
- **Sequence modelling:** Builds an LSTM regressor (Keras/TensorFlow) on scaled close prices with configurable lookback windows.
- **Composite confidence:** Normalizes indicator readings, sentiment, and the next-step price prediction into a single float that behaves well even when indicators are missing.
- **Lightweight evaluation:** When run as a script the module prints MAE/RMSE, directional accuracy, and an average confidence score across the hold-out period.

## Repository Layout
- `my_trading_model.py` — main pipeline with reusable functions and an optional example workflow under `if __name__ == "__main__"`.
- `talib_compat.py` — minimal pandas-based replacements for the TA-Lib API surface used here.
- `tests_trading_model.py` — unittest suite that exercises scraping, preprocessing, modelling, and scoring helpers (designed for `pytest`).
- `requirements.txt` — pinned runtime dependencies used by the workflow and tests.
- `python-app.yml` — GitHub Actions recipe for linting with Flake8 and running the tests on pushes and pull requests.

## Getting Started
1. Use Python 3.10+ and create a virtual environment if possible.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note:* `TA-Lib` can be tricky to compile; the project continues to work with the included compatibility layer if the wheel fails to install.

## Running the Workflow
Fed with real market data and news, the example routine performs every stage end-to-end:
```bash
python my_trading_model.py
```
What to expect:
- Downloads the ticker specified by `ticker_sym` (`AAPL` by default) for the configured date range.
- Cleans and augments the price frame with trend and momentum indicators.
- Scrapes current Yahoo Finance headlines and derives sentiment scores.
- Trains an LSTM on a 60-day lookback window and predicts the next close.
- Prints evaluation metrics plus the composite confidence score.

Because the workflow touches external APIs, ensure outbound HTTP access is permitted before running it.

### Example Output
```
Confidence Score for the stock: 1.0001876419031843
Out-of-sample MAE: 3.8313  RMSE: 4.5361  Directional accuracy: 60.8%
Average Confidence Score (test period): 0.657

=== Summary ===
Prediction Accuracy: Around 61%
Confidence Score: Averaged at 0.66
```
These numbers come from the default AAPL run spanning 2020-01-01 to 2021-01-01.

## Reading The Output
- **Confidence Score for the stock** — Single snapshot combining indicators, sentiment, and the next-step LSTM forecast. Values stay between 0 and 1; numbers near 1 mean every input tilts bullish right now, while values near 0 flag a bearish or uncertain regime.
- **Out-of-sample MAE / RMSE** — Average absolute and squared price errors (in dollars) on the hold-out set. Lower is better; compare them to the share price to judge percentage error.
- **Directional accuracy** — Share of test predictions that guessed the next day’s up/down move correctly. Around 50% is coin-flip; sustained improvement above that threshold signals real edge.

## Testing
The suite targets the pure-Python functions so it is fast to run locally:
```bash
pytest
```
Tests skip the `yfinance` download check automatically when the package is missing. Network calls are mocked so no live HTTP requests are made during testing.

## Design Notes
- Functions are individually importable, making it easy to plug just the pieces needed for other projects.
- Heavy operations and optional dependencies are fenced off from import-time execution, which keeps unit tests and notebooks responsive.
- The confidence metric clamps missing technical fields and neutral sentiment to sensible defaults, keeping the score stable in sparse data scenarios.
- `talib_compat.py` intentionally mirrors the TA-Lib API naming so swapping in the native library remains a one-line change.

## Roadmap Ideas
- Stream live prices and headlines to drive intraday signals.
- Persist trained models and scalers for reuse outside the demo script.
- Expose the workflow through a lightweight REST or Streamlit interface.
- Expand sentiment sources to include earnings call transcripts and social feeds.

> This repository is a research sandbox. It is **not** investment advice, and no performance guarantees are implied.
