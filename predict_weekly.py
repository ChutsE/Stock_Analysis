"""
LSTM Model Inference Script.
Loads trained model and predicts stock prices week by week (7 days ahead).
Supports both CSV files and live data from Yahoo Finance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import timedelta
import argparse
import sys

# Import stock data extractor functions
try:
    from stock_data_extractor import get_historical_data
except ImportError:
    print("Warning: stock_data_extractor.py not found in current directory.")
    get_historical_data = None


def load_close_data(csv_file):
    """
    Load Close price data from historical CSV.
    
    Args:
        csv_file (Path): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with Date and Close columns
    """
    try:
        data = pd.read_csv(csv_file)
        
        # Find date column
        date_col = None
        if "Date" in data.columns:
            date_col = "Date"
        elif data.columns[0].startswith("Unnamed"):
            date_col = data.columns[0]
            data = data.rename(columns={date_col: "Date"})
            date_col = "Date"
        
        if date_col is None or "Close" not in data.columns:
            print(f"Error: {csv_file.name} missing Date or Close column")
            return None
        
        # Convert to datetime and sort
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date", "Close"])
        data = data.sort_values("Date").reset_index(drop=True)
        
        return data[["Date", "Close"]]
    
    except Exception as e:
        print(f"Error loading {csv_file.name}: {e}")
        return None


def load_live_data(ticker, lookback=60):
    """
    Load live historical data directly from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        lookback (int): Number of days needed (adds buffer for safety)
        
    Returns:
        pd.DataFrame: DataFrame with Date and Close columns
    """
    if get_historical_data is None:
        print("Error: stock_data_extractor module not available.")
        return None
    
    try:
        print(f"Fetching live data for {ticker}...")
        # Request more data than needed to ensure we have enough
        period_days = max(lookback + 30, 100)
        hist = get_historical_data(ticker, period=f"{period_days}d", interval="1d")
        
        if hist is None or hist.empty:
            print(f"No data returned for {ticker}")
            return None
        
        # Reset index to get Date as column
        data = hist.reset_index()
        
        if "Date" not in data.columns:
            print("Error: Date column not found in live data")
            return None
        
        if "Close" not in data.columns:
            print("Error: Close column not found in live data")
            return None
        
        # Clean and sort
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date", "Close"])
        data = data.sort_values("Date").reset_index(drop=True)
        
        print(f"Downloaded {len(data)} records")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        return data[["Date", "Close"]]
    
    except Exception as e:
        print(f"Error fetching live data for {ticker}: {e}")
        return None


def predict_next_n_days(model, last_sequence, scaler, n_days=7):
    """
    Predict next N days using the trained model.
    Uses rolling prediction (each prediction feeds into the next).
    
    Args:
        model: Trained Keras model
        last_sequence (np.array): Last lookback period data (normalized)
        scaler: MinMaxScaler used for training
        n_days (int): Number of days to predict
        
    Returns:
        np.array: Predicted prices (original scale)
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_days):
        # Reshape for prediction (1, lookback, 1)
        current_input = current_sequence.reshape(1, len(current_sequence), 1)
        
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence: remove first value, add prediction
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Convert to array and inverse transform
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()


def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(
        description="LSTM model inference for weekly stock price prediction."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (.h5 file). If not provided, searches in data folder.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to historical CSV file. If not provided, searches in data folder.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to predict (default: 7 for one week)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live data from Yahoo Finance instead of using CSV",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Ticker symbol (required when using --live)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LSTM MODEL INFERENCE - WEEKLY STOCK PREDICTION")
    print("=" * 70)
    
    data_models_dir = Path("data/models/")
    data_historical_dir = Path("data/historical/")
    lookback = 60  # Must match training configuration
    
    # Find model file
    if args.model:
        model_path = Path(args.model)
    else:
        model_files = list(data_models_dir.glob("*_lstm_model.h5"))
        if not model_files:
            print("\nError: No trained model found in data folder.")
            print("Train a model first using train_lstm_model.py")
            return
        model_path = model_files[0]
        print(f"\nFound model: {model_path.name}")
    
    if not model_path.exists():
        print(f"\nError: Model file not found: {model_path}")
        return
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load historical data (either from CSV or live)
    if args.live:
        # Live mode: fetch from Yahoo Finance
        if not args.ticker:
            print("\nError: --ticker is required when using --live mode")
            print("Example: python predict_weekly.py --live --ticker IVVPESO.MX")
            return
        
        print(f"\nMode: LIVE DATA")
        print(f"Ticker: {args.ticker}")
        data = load_live_data(args.ticker, lookback)
        if data is None:
            return
    else:
        # CSV mode: load from file
        print(f"\nMode: CSV FILE")
        
        if args.csv:
            csv_file = Path(args.csv)
        else:
            # Extract ticker from model name
            model_stem = model_path.stem.replace("_lstm_model", "")
            csv_pattern = f"{model_stem}.csv"
            csv_files = list(data_historical_dir.glob(csv_pattern))
            
            if not csv_files:
                # Fallback: find any historical CSV
                csv_files = list(data_historical_dir.glob("*_historical_*.csv"))
            
            if not csv_files:
                print("\nError: No historical CSV found.")
                print("Tip: Use --live --ticker SYMBOL to fetch data directly.")
                return
            
            csv_file = csv_files[0]
            print(f"Using CSV: {csv_file.name}")
        
        if not csv_file.exists():
            print(f"\nError: CSV file not found: {csv_file}")
            return
        
        print(f"\nLoading historical data from: {csv_file.name}")
        data = load_close_data(csv_file)
        if data is None:
            return
    
    print(f"Total records: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
    
    # Check if we have enough data
    if len(data) < lookback:
        print(f"\nError: Need at least {lookback} days of data. Found: {len(data)}")
        return
    
    # Prepare data for prediction
    # Use all historical data to fit scaler (consistent with training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data_scaled = scaler.fit_transform(data[['Close']].values)
    
    # Get last lookback days for prediction
    last_sequence = all_data_scaled[-lookback:]
    
    # Make predictions
    print(f"\nPredicting next {args.days} days...")
    predictions = predict_next_n_days(model, last_sequence, scaler, args.days)
    
    # Create prediction dates (business days)
    last_date = data['Date'].iloc[-1]
    prediction_dates = []
    current_date = last_date
    
    for i in range(args.days):
        current_date = current_date + timedelta(days=1)
        # Skip weekends (simple approach)
        while current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            current_date = current_date + timedelta(days=1)
        prediction_dates.append(current_date)
    
    # Display predictions
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    print(f"{'Date':<15} {'Predicted Close':<20} {'Change from Last':<20}")
    print("-" * 70)
    
    last_close = data['Close'].iloc[-1]
    for date, pred_price in zip(prediction_dates, predictions):
        change = pred_price - last_close
        change_pct = (change / last_close) * 100
        print(f"{date.strftime('%Y-%m-%d'):<15} ${pred_price:<18.2f} ${change:+.2f} ({change_pct:+.2f}%)")
        last_close = pred_price
    
    total_change = predictions[-1] - data['Close'].iloc[-1]
    total_change_pct = (total_change / data['Close'].iloc[-1]) * 100
    print("-" * 70)
    print(f"{'Week Total:':<15} ${predictions[-1]:<18.2f} ${total_change:+.2f} ({total_change_pct:+.2f}%)")
    
    # Plot predictions
    print("\nGenerating visualization...")
    
    # Show last 30 days + predictions (reduced for better prediction visibility)
    lookback_display = min(30, len(data))
    recent_data = data.iloc[-lookback_display:]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data with blue color and blue markers
    ax.plot(recent_data['Date'], recent_data['Close'], 
            label='Historical', linewidth=2, color='blue', marker='o', 
            markersize=4, markerfacecolor='blue', markeredgecolor='blue')
    
    # Plot predictions
    all_dates = list(recent_data['Date']) + prediction_dates
    all_prices = list(recent_data['Close']) + list(predictions)
    
    # Connect last historical point to first prediction
    transition_dates = [recent_data['Date'].iloc[-1]] + prediction_dates
    transition_prices = [recent_data['Close'].iloc[-1]] + list(predictions)
    
    ax.plot(transition_dates, transition_prices, 
            label='Predicted (Next Week)', linewidth=2, color='red', 
            linestyle='--', marker='o', markersize=5)
    
    # Add vertical line at prediction start
    ax.axvline(x=recent_data['Date'].iloc[-1], color='gray', 
               linestyle=':', alpha=0.7, label='Prediction Start')
    
    # Create blinking point for current price (last historical point)
    current_date = recent_data['Date'].iloc[-1]
    current_price = recent_data['Close'].iloc[-1]
    blinking_point, = ax.plot([current_date], [current_price], 
                              marker='o', markersize=15, 
                              color='blue', markeredgecolor='white', 
                              markeredgewidth=2, zorder=5)
    
    # Formatting
    ax.set_title(f'{args.ticker} Stock Price - Next {args.days} Days (Current: ${current_price:.2f})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Animation function for blinking effect
    def animate(frame):
        # Blink between visible and invisible (0.5 second intervals)
        if frame % 2 == 0:
            blinking_point.set_alpha(1.0)
            blinking_point.set_markersize(15)
        else:
            blinking_point.set_alpha(0.3)
            blinking_point.set_markersize(13)
        return blinking_point,
    
    # Create animation (blinks every 500ms)
    anim = animation.FuncAnimation(fig, animate, frames=100, 
                                   interval=500, blit=True, repeat=True)
    
    plt.show()
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
