# 📈 Stock Analysis & LSTM Price Prediction

Deep learning toolkit for stock price prediction using LSTM neural networks. Download data, train models, and forecast future prices with live visualizations.

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download historical data:**
```bash
python save_history_stock.py AAPL --period 10y
```

3. **Train LSTM model:**
```bash
python train_lstm_model.py
```

4. **Predict next week:**
```bash
python predict_weekly.py --live --ticker AAPL --days 7
```

## 📊 Scripts Usage

### 1. Download Stock Data
```bash
python save_history_stock.py <TICKER> [--period PERIOD] [--interval INTERVAL]
```
**Examples:**
```bash
python save_history_stock.py AAPL --period 10y
python save_history_stock.py TSLA --period 5y --interval 1d
python save_history_stock.py IVVPESO.MX --period 10y
```

### 2. Visualize Historical Data
```bash
python visualize_stocks.py
```
Shows charts for all CSV files in `data/` folder.

### 3. Train LSTM Model
```bash
python train_lstm_model.py
```
Trains on historical data with last year as validation. Model saved to `data/models/`.

### 4. Predict Single Stock
```bash
python predict_weekly.py --live --ticker <TICKER> [--days N]
```
**Examples:**
```bash
python predict_weekly.py --live --ticker AAPL --days 7
python predict_weekly.py --live --ticker MSFT --days 14
python predict_weekly.py --live --ticker OXY
```

### 5. Monitor Multiple Stocks (NEW! ⭐)
```bash
python monitor_stocks.py --tickers <TICKER1> <TICKER2> <TICKER3> [--days N]
```
**Examples:**
```bash
python monitor_stocks.py --tickers AAPL MSFT TSLA
python monitor_stocks.py --tickers OXY IVVPESO.MX SPY --days 7
python monitor_stocks.py --tickers AAPL GOOGL AMZN TSLA NVDA
```
Opens separate windows for each stock with live predictions and animated current price indicator.

## 🎯 Features

- ✅ Download historical data from Yahoo Finance (stocks, ETFs, crypto, forex)
- ✅ Train LSTM neural networks on historical prices
- ✅ Predict future prices (7-30 days ahead)
- ✅ Live data fetching without pre-saved CSVs
- ✅ **Multi-stock monitoring with multiprocessing**
- ✅ Animated visualizations with blinking current price
- ✅ Business days handling (skips weekends)

## 📦 Project Structure

```
Stock_Analysis/
├── data/
│   ├── historical/              # Historical CSV files
│   ├── models/                  # Trained LSTM models (.h5)
│   └── info/                    # Ticker info logs
├── save_history_stock.py        # Download data
├── visualize_stocks.py          # Plot historical data
├── train_lstm_model.py          # Train LSTM model
├── predict_weekly.py            # Single stock prediction
├── monitor_stocks.py            # Multi-stock monitor (NEW!)
├── stock_data_extractor.py      # Live data utilities
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.13+
- yfinance 0.2.28+
- pandas 1.5.0+
- scikit-learn 1.3.0+
- matplotlib 3.7.0+
- numpy 1.24.0+

Install all with: `pip install -r requirements.txt`

## 📝 Notes

- **Data Source**: Yahoo Finance (free, but rate-limited)
- **Supported Instruments**: Stocks (AAPL), ETFs (SPY), Indexes (^GSPC), Crypto (BTC-USD), Forex (EURUSD=X)
- **Mexican Securities**: Add `.MX` suffix (e.g., `IVVPESO.MX`)
- **Prediction Method**: Rolling LSTM predictions using 60-day lookback window
- **Disclaimer**: For educational purposes only. Not financial advice.

## 🎓 Complete Workflow Example

```bash
# Download 10 years of Apple data
python save_history_stock.py AAPL --period 10y

# Train model on the data
python train_lstm_model.py

# Predict next 7 days for Apple
python predict_weekly.py --live --ticker AAPL --days 7

# Monitor multiple tech stocks
python monitor_stocks.py --tickers AAPL MSFT GOOGL AMZN
```

## 🌟 What's New

- **monitor_stocks.py**: Monitor multiple stocks in parallel using multiprocessing
- Each stock gets its own window with independent predictions
- Animated blinking indicator for current price
- Blue line with markers for historical data
- Red dashed line for predictions

## 🤝 Contributing

Contributions welcome! Submit issues or pull requests.

---

**Last Updated**: March 2026