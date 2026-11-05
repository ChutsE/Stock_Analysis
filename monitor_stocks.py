"""
Multi-Stock Monitor with LSTM Predictions
Monitors multiple stocks simultaneously using multiprocessing.
Each stock gets its own window with live predictions and blinking current price.
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
import multiprocessing
import yfinance as yf


def get_historical_data(ticker, period="1mo", interval="1d"):
    """
    Get historical stock data
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
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
    try:
        print(f"[{ticker}] Fetching live data...")
        period_days = max(lookback + 30, 100)
        hist = get_historical_data(ticker, period=f"{period_days}d", interval="1d")
        
        if hist is None or hist.empty:
            print(f"[{ticker}] No data returned")
            return None
        
        data = hist.reset_index()
        
        if "Date" not in data.columns or "Close" not in data.columns:
            print(f"[{ticker}] Missing Date or Close column")
            return None
        
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date", "Close"])
        data = data.sort_values("Date").reset_index(drop=True)
        
        print(f"[{ticker}] Downloaded {len(data)} records")
        print(f"[{ticker}] Latest close: ${data['Close'].iloc[-1]:.2f}")
        
        return data[["Date", "Close"]]
    
    except Exception as e:
        print(f"[{ticker}] Error fetching data: {e}")
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
        current_input = current_sequence.reshape(1, len(current_sequence), 1)
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()


def get_trading_dates_after(start_date, n_days):
    """Return the next N weekdays after a date."""
    dates = []
    current_date = start_date

    while len(dates) < n_days:
        current_date = current_date + timedelta(days=1)
        if current_date.weekday() < 5:
            dates.append(current_date)

    return dates


def monitor_single_stock(ticker, model_path, days=7, lookback=60, past_days=0):
    """
    Monitor a single stock in its own process/window.
    
    Args:
        ticker (str): Stock ticker symbol
        model_path (Path): Path to trained LSTM model
        days (int): Number of days to predict
        lookback (int): Lookback period for model input
        past_days (int): Number of previous historical days to use as prediction anchors
    """
    try:
        # Set process title
        multiprocessing.current_process().name = f"Monitor-{ticker}"
        
        print(f"\n{'='*60}")
        print(f"MONITORING: {ticker}")
        print(f"{'='*60}")
        
        # Load model
        print(f"[{ticker}] Loading model...")
        model = tf.keras.models.load_model(model_path)
        
        # Load live data
        data = load_live_data(ticker, lookback + past_days)
        minimum_records = lookback + past_days
        if data is None or len(data) < minimum_records:
            print(f"[{ticker}] Insufficient data. Need at least {minimum_records} days.")
            return
        
        # Prepare data
        scaler = MinMaxScaler(feature_range=(0, 1))
        all_data_scaled = scaler.fit_transform(data[['Close']].values)
        # Generate one forecast for each requested historical anchor, including today.
        anchor_start = len(data) - 1 - past_days
        forecasts = []
        for anchor_index in range(anchor_start, len(data)):
            last_sequence = all_data_scaled[anchor_index - lookback + 1:anchor_index + 1]
            predictions = predict_next_n_days(model, last_sequence, scaler, days)
            anchor_date = data['Date'].iloc[anchor_index]
            forecasts.append({
                "anchor_date": anchor_date,
                "dates": get_trading_dates_after(anchor_date, days),
                "prices": predictions,
            })

        current_forecast = forecasts[-1]
        predictions = current_forecast["prices"]
        prediction_dates = current_forecast["dates"]

        print(f"[{ticker}] Predicting {days} days from {len(forecasts)} anchor(s)...")
        
        # Display predictions
        print(f"\n[{ticker}] PREDICTIONS:")
        print(f"{'Date':<15} {'Price':<15} {'Change':<15}")
        print("-" * 45)
        last_close = data['Close'].iloc[-1]
        for date, pred_price in zip(prediction_dates, predictions):
            change = pred_price - last_close
            change_pct = (change / last_close) * 100
            print(f"{date.strftime('%Y-%m-%d'):<15} ${pred_price:<13.2f} {change_pct:+.2f}%")
            last_close = pred_price
        
        # Create visualization
        lookback_display = min(max(30, past_days + 1), len(data))
        recent_data = data.iloc[-lookback_display:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data with blue markers
        ax.plot(recent_data['Date'], recent_data['Close'], 
                label='Historical', linewidth=2, color='blue', marker='o', 
                markersize=4, markerfacecolor='blue', markeredgecolor='blue')
        
        # Plot predictions
        for forecast_index, forecast in enumerate(forecasts):
            transition_dates = [forecast["anchor_date"]] + forecast["dates"]
            transition_prices = [
                data.loc[data['Date'] == forecast["anchor_date"], 'Close'].iloc[0]
            ] + list(forecast["prices"])
            is_current = forecast_index == len(forecasts) - 1
            label = (
                'Predicted from current'
                if is_current
                else 'Predicted from past day' if forecast_index == 0 else None
            )
            ax.plot(
                transition_dates,
                transition_prices,
                label=label,
                linewidth=2 if is_current else 1.2,
                color='red' if is_current else 'darkorange',
                alpha=1.0 if is_current else 0.55,
                linestyle='--',
                marker='o' if is_current else None,
                markersize=5,
            )
        
        # Add vertical line at prediction start
        ax.axvline(x=recent_data['Date'].iloc[-1], color='gray', 
                   linestyle=':', alpha=0.7, label='Prediction Start')
        
        # Create blinking point for current price
        current_date = recent_data['Date'].iloc[-1]
        current_price = recent_data['Close'].iloc[-1]
        blinking_point, = ax.plot([current_date], [current_price], 
                                  marker='o', markersize=15, 
                                  color='blue', markeredgecolor='white', 
                                  markeredgewidth=2, zorder=5)
        
        # Formatting
        total_change = predictions[-1] - data['Close'].iloc[-1]
        total_change_pct = (total_change / data['Close'].iloc[-1]) * 100
        
        ax.set_title(f'{ticker} - {days} Days x {len(forecasts)} Anchors | Current: ${current_price:.2f} | Predicted: ${predictions[-1]:.2f} ({total_change_pct:+.2f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Close Price ($)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Animation function for blinking effect
        def animate(frame):
            if frame % 2 == 0:
                blinking_point.set_alpha(1.0)
                blinking_point.set_markersize(15)
            else:
                blinking_point.set_alpha(0.3)
                blinking_point.set_markersize(13)
            return blinking_point,
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=100, 
                                       interval=500, blit=True, repeat=True)
        
        print(f"[{ticker}] Opening visualization window...")
        plt.show()
        
    except Exception as e:
        print(f"[{ticker}] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to launch multi-stock monitor."""
    
    parser = argparse.ArgumentParser(
        description="Monitor multiple stocks with LSTM predictions simultaneously."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs='+',
        required=True,
        help="Stock ticker symbols (space-separated). Example: --tickers AAPL MSFT TSLA"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (.h5 file). If not provided, searches in data/models folder."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to predict (default: 7)"
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=0,
        help="Number of previous days to use as additional prediction anchors (default: 0)"
    )
    
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.past_days < 0:
        parser.error("--past-days cannot be negative")
    
    print("=" * 70)
    print("MULTI-STOCK MONITOR WITH LSTM PREDICTIONS")
    print("=" * 70)
    print(f"\nMonitoring {len(args.tickers)} stocks: {', '.join(args.tickers)}")
    print(f"Prediction window: {args.days} days from {args.past_days + 1} anchor(s)")
    
    # Find model file
    data_models_dir = Path("data/models/")
    if args.model:
        model_path = Path(args.model)
    else:
        model_files = list(data_models_dir.glob("*_lstm_model.h5"))
        if not model_files:
            print("\nError: No trained model found in data/models folder.")
            print("Train a model first using train_lstm_model.py")
            return
        model_path = model_files[0]
        print(f"\nUsing model: {model_path.name}")
    
    if not model_path.exists():
        print(f"\nError: Model file not found: {model_path}")
        return
    
    lookback = 60  # Must match training configuration
    
    # Create a process for each ticker
    processes = []
    
    print(f"\nLaunching {len(args.tickers)} monitoring windows...")
    print("Note: Each stock opens in a separate window.")
    print("Close all windows to exit.\n")
    
    for ticker in args.tickers:
        p = multiprocessing.Process(
            target=monitor_single_stock,
            args=(ticker, model_path, args.days, lookback, args.past_days),
            name=f"Monitor-{ticker}"
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll monitoring windows closed. Exiting.")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
