# Stock Analysis & LSTM Price Prediction

A comprehensive stock market analysis toolkit with deep learning capabilities for price prediction. This project enables users to download historical market data, visualize price trends, train LSTM neural networks, and forecast future stock prices for any publicly traded security.

## Description

This system combines financial data extraction with machine learning to predict stock prices. Using Yahoo Finance as the data source, the toolkit supports stocks, ETFs, indexes, forex, and cryptocurrencies. The LSTM (Long Short-Term Memory) neural network architecture learns patterns from historical closing prices to generate week-ahead predictions. The modular design allows users to work with any ticker symbol, customize training parameters, and make predictions using either saved data or live market feeds.

Key capabilities include automated data collection with customizable periods (1 day to 10+ years), technical indicator calculations (moving averages, rolling max/min), model training with validation splits, and rolling predictions for future price movements. The system generates comprehensive visualizations for both historical analysis and prediction evaluation.

## Features

- **Data Collection**: Download historical OHLCV data for any ticker from Yahoo Finance
- **Visualization**: Plot price charts with volume for historical analysis
- **LSTM Training**: Train deep learning models on historical close prices with automatic train/validation split
- **Price Prediction**: Forecast stock prices week-by-week with trained models
- **Live Data**: Support for real-time data fetching without pre-saved CSVs
- **Flexible Configuration**: Customizable periods, intervals, and prediction windows

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Historical Data
```bash
python save_advanced_csv.py IVVPESO.MX --period 10y --interval 1d
```

### 2. Visualize Data
```bash
python plot_historical_data.py
```

### 3. Train LSTM Model
```bash
python train_lstm_model.py
```

### 4. Make Predictions
```bash
# Using saved CSV
python predict_weekly.py

# Using live data
python predict_weekly.py --live --ticker IVVPESO.MX --days 7
```

## Scripts

### `save_advanced_csv.py`
Downloads historical market data and saves to CSV with optional statistics.

**Usage:**
```bash
python save_advanced_csv.py <TICKER> [--period PERIOD] [--interval INTERVAL] [--stats]
```

**Examples:**
```bash
# Basic download
python save_advanced_csv.py AAPL

# 5-year weekly data
python save_advanced_csv.py ^GSPC --period 5y --interval 1wk

# With statistics (MA, returns, max/min)
python save_advanced_csv.py SPY --period 2y --stats
```

### `plot_historical_data.py`
Visualizes historical price data from CSV files.

**Usage:**
```bash
python plot_historical_data.py
```

Automatically reads all `*_historical_*.csv` files from the `data/` folder and generates OHLC + Volume charts.

### `train_lstm_model.py`
Trains LSTM neural network model for price prediction.

**Usage:**
```bash
python train_lstm_model.py
```

**Configuration:**
- Lookback window: 60 days
- Validation split: Last 365 days (1 year)
- Architecture: 2 LSTM layers (50 units each) + Dropout + Dense
- Training: 50 epochs, batch size 32

**Outputs:**
- Trained model: `data/*_lstm_model.h5`
- Training metrics: RMSE, MAE
- Visualization: Loss curves and prediction vs actual charts

### `predict_weekly.py`
Performs inference using trained model to predict future prices.

**Usage:**
```bash
# CSV mode
python predict_weekly.py [--model PATH] [--csv PATH] [--days N]

# Live mode
python predict_weekly.py --live --ticker SYMBOL [--days N]
```

**Examples:**
```bash
# Predict next 7 days from CSV
python predict_weekly.py

# Predict next 14 days with live data
python predict_weekly.py --live --ticker IVVPESO.MX --days 14

# Custom model and CSV
python predict_weekly.py --model data/AAPL_lstm_model.h5 --csv data/AAPL_historical_1y_20260307.csv
```

## Supported Instruments

- **Stocks**: AAPL, MSFT, TSLA, etc.
- **ETFs**: IVV, SPY, QQQ, IVVPESO.MX
- **Indexes**: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
- **Forex**: EURUSD=X, GBPUSD=X
- **Crypto**: BTC-USD, ETH-USD

## Finding Tickers

- **Yahoo Finance**: https://finance.yahoo.com
- **TradingView**: https://www.tradingview.com/symbols/
- **BMV (Mexico)**: https://www.bmv.com.mx

**Tip**: Mexican securities usually require `.MX` suffix (e.g., `IVVPESO.MX`)

## Project Structure

```
Stock_Analysis/
├── data/                          # Generated data folder
│   ├── *_historical_*.csv        # Historical OHLCV data
│   ├── *_with_stats_*.csv        # Data with technical indicators
│   ├── *_info_*.log              # Ticker information logs
│   └── *_lstm_model.h5           # Trained models
├── save_advanced_csv.py          # Data downloader
├── plot_historical_data.py       # Visualization tool
├── train_lstm_model.py           # Model training
├── predict_weekly.py             # Prediction/inference
├── stock_data_extractor.py       # Live data utilities
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Workflow Example

Complete workflow for predicting IVVPESO.MX:

```bash
# Step 1: Download 10 years of data
python save_advanced_csv.py IVVPESO.MX --period 10y

# Step 2: Visualize the data
python plot_historical_data.py

# Step 3: Train LSTM model (uses last year as validation)
python train_lstm_model.py

# Step 4: Predict next week
python predict_weekly.py --live --ticker IVVPESO.MX --days 7
```

## Model Details

### Architecture
- Input: 60-day sliding window of normalized closing prices
- Layer 1: LSTM(50 units, return_sequences=True)
- Dropout: 0.2
- Layer 2: LSTM(50 units)
- Dropout: 0.2
- Dense: 25 units
- Output: 1 unit (next day price)

### Training Strategy
- Loss: Mean Squared Error
- Optimizer: Adam
- Normalization: MinMaxScaler (0-1 range)
- Validation: Last 365 days reserved for testing

### Prediction Method
- Rolling predictions: Each prediction feeds into the next
- Week-ahead forecast: 7 consecutive daily predictions
- Business days only: Automatically skips weekends

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- yfinance 0.2.28+
- pandas 1.5.0+
- scikit-learn 1.3.0+
- matplotlib 3.7.0+
- numpy 1.24.0+

## Notes

- **Data Source**: Yahoo Finance API (free but with rate limits)
- **Market Hours**: Real-time data available during trading hours; outside hours shows last close
- **Prediction Accuracy**: LSTM predictions are probabilistic; actual results may vary
- **Risk Disclaimer**: This tool is for educational/research purposes. Not financial advice.

## License

This project is open source and available for educational purposes.

## Contributing

Contributions welcome! Feel free to submit issues or pull requests.

---

**Last Updated**: March 2026
