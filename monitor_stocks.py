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
from html import escape


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
        
        columns = ["Date", "Close"]
        for column in ("High", "Low"):
            if column in data.columns:
                columns.append(column)
        return data[columns]
    
    except Exception as e:
        print(f"[{ticker}] Error fetching data: {e}")
        return None


def add_technical_indicators(data, rsi_period=14):
    """Add RSI, moving averages, and ATR to a price DataFrame."""
    enriched = data.copy()
    changes = enriched["Close"].diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    enriched["RSI"] = 100 - (100 / (1 + relative_strength))
    enriched.loc[(average_loss == 0) & (average_gain > 0), "RSI"] = 100
    enriched["SMA20"] = enriched["Close"].rolling(20).mean()
    enriched["SMA50"] = enriched["Close"].rolling(50).mean()

    if {"High", "Low"}.issubset(enriched.columns):
        previous_close = enriched["Close"].shift(1)
        true_range = pd.concat(
            [
                enriched["High"] - enriched["Low"],
                (enriched["High"] - previous_close).abs(),
                (enriched["Low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        enriched["ATR14"] = true_range.rolling(14).mean()
    else:
        enriched["ATR14"] = enriched["Close"].diff().abs().rolling(14).mean()

    return enriched


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


def build_forecast(ticker, model_path, days=7, lookback=60, past_days=0):
    """Load a ticker model and return data, forecasts, and current prediction."""
    print(f"[{ticker}] Loading model...")
    model = tf.keras.models.load_model(model_path)
    data = load_live_data(ticker, lookback + past_days)
    minimum_records = lookback + past_days
    if data is None or len(data) < minimum_records:
        raise ValueError(f"Insufficient data. Need at least {minimum_records} days.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data_scaled = scaler.fit_transform(data[['Close']].values)
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

    return add_technical_indicators(data), forecasts


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
        
        data, forecasts = build_forecast(
            ticker, model_path, days, lookback, past_days
        )

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


def find_model_for_ticker(ticker, models_dir):
    """Find the newest model whose filename starts with the exact ticker."""
    matching_models = sorted(
        models_dir.glob(f"{ticker}_historical_*_lstm_model.h5"),
        reverse=True,
    )
    return matching_models[0] if matching_models else None


def evaluate_signal(data, predicted_price, rsi_oversold=30, rsi_overbought=70):
    """Return a mechanical buy, sell, or wait signal from technical evidence."""
    latest = data.dropna(subset=["RSI", "SMA20", "SMA50", "ATR14"]).iloc[-1]
    current_price = float(latest["Close"])
    predicted_return = (float(predicted_price) / current_price - 1) * 100
    buy_triggers = []
    sell_triggers = []

    if latest["RSI"] <= rsi_oversold:
        buy_triggers.append("RSI <= 30")
    if latest["RSI"] >= rsi_overbought:
        sell_triggers.append("RSI >= 70")
    if current_price > latest["SMA20"]:
        buy_triggers.append("Precio > SMA20")
    else:
        sell_triggers.append("Precio <= SMA20")
    if latest["SMA20"] > latest["SMA50"]:
        buy_triggers.append("SMA20 > SMA50")
    else:
        sell_triggers.append("SMA20 <= SMA50")
    if predicted_return >= 2:
        buy_triggers.append("Pronóstico >= +2%")
    elif predicted_return <= -2:
        sell_triggers.append("Pronóstico <= -2%")

    buy_score = len(buy_triggers)
    sell_score = len(sell_triggers)
    if buy_score >= 3 and buy_score > sell_score:
        decision = "COMPRA"
    elif sell_score >= 3 and sell_score > buy_score:
        decision = "VENTA"
    else:
        decision = "ESPERA"

    return {
        "decision": decision,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "confidence": max(buy_score, sell_score) / 4 * 100,
        "buy_triggers": buy_triggers,
        "sell_triggers": sell_triggers,
        "predicted_return": predicted_return,
    }


def create_dashboard(tickers, model_paths, days, lookback, past_days, output_path):
    """Generate a standalone HTML dashboard for all requested tickers."""
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots

    cards = []
    failed = []
    for ticker in tickers:
        try:
            data, forecasts = build_forecast(
                ticker, model_paths[ticker], days, lookback, past_days
            )
            current_forecast = forecasts[-1]
            current_price = float(data['Close'].iloc[-1])
            predicted_price = float(current_forecast['prices'][-1])
            change_pct = (predicted_price / current_price - 1) * 100
            signal = evaluate_signal(data, predicted_price)
            recent_data = data.iloc[-min(max(30, past_days + 1), len(data)):]

            figure = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=("Precio y tendencia", "RSI (14)")
            )
            figure.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['Close'],
                mode='lines+markers',
                name='Histórico',
                line={'color': '#2563eb', 'width': 2},
            ), row=1, col=1)
            for column, color in (("SMA20", "#16a34a"), ("SMA50", "#9333ea")):
                figure.add_trace(go.Scatter(
                    x=recent_data['Date'],
                    y=recent_data[column],
                    mode='lines',
                    name=column,
                    line={'color': color, 'width': 1.5},
                ), row=1, col=1)
            for forecast_index, forecast in enumerate(forecasts):
                anchor_price = float(data.loc[
                    data['Date'] == forecast['anchor_date'], 'Close'
                ].iloc[0])
                is_current = forecast_index == len(forecasts) - 1
                figure.add_trace(go.Scatter(
                    x=[forecast['anchor_date']] + forecast['dates'],
                    y=[anchor_price] + list(forecast['prices']),
                    mode='lines+markers' if is_current else 'lines',
                    name='Desde hoy' if is_current else 'Ancla histórica',
                    line={
                        'color': '#dc2626' if is_current else '#f59e0b',
                        'width': 2 if is_current else 1,
                        'dash': 'dash',
                    },
                    opacity=1 if is_current else 0.45,
                    showlegend=forecast_index in (0, len(forecasts) - 1),
                ), row=1, col=1)
            figure.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data['RSI'],
                mode='lines',
                name='RSI14',
                line={'color': '#ea580c', 'width': 2},
            ), row=2, col=1)
            figure.add_hline(y=70, line_dash='dash', line_color='#b91c1c', row=2, col=1)
            figure.add_hline(y=30, line_dash='dash', line_color='#15803d', row=2, col=1)
            figure.update_layout(
                height=500,
                margin={'l': 45, 'r': 20, 't': 20, 'b': 45},
                hovermode='x unified',
                template='plotly_white',
                legend={'orientation': 'h', 'y': 1.08},
            )
            figure.update_yaxes(title_text='Precio', row=1, col=1)
            figure.update_yaxes(title_text='RSI', range=[0, 100], row=2, col=1)
            figure.update_xaxes(title_text='Fecha', row=2, col=1)
            chart = plot(
                figure,
                output_type='div',
                include_plotlyjs='inline' if not cards else False,
                config={'responsive': True, 'displaylogo': False},
            )
            rows = ''.join(
                f"<tr><td>{date.strftime('%Y-%m-%d')}</td>"
                f"<td>${price:.2f}</td></tr>"
                for date, price in zip(
                    current_forecast['dates'], current_forecast['prices']
                )
            )
            direction = 'positive' if change_pct >= 0 else 'negative'
            signal_class = signal['decision'].lower()
            trigger_text = '; '.join(
                signal['buy_triggers'] if signal['decision'] == 'COMPRA'
                else signal['sell_triggers'] if signal['decision'] == 'VENTA'
                else signal['buy_triggers'] + signal['sell_triggers']
            ) or 'Sin triggers dominantes'
            latest_rsi = float(data['RSI'].iloc[-1])
            latest_atr = float(data['ATR14'].iloc[-1])
            rsi_state = (
                'Sobrecomprado' if latest_rsi >= 70
                else 'Sobrevendido' if latest_rsi <= 30
                else 'Neutral'
            )
            cards.append(f"""
                <article class="card">
                  <div class="card-head">
                    <div><h2>{escape(ticker)}</h2>
                    <span class="muted">Modelo: {escape(model_paths[ticker].stem)}</span></div>
                    <div class="price"><strong>${current_price:.2f}</strong>
                    <span class="{direction}">{change_pct:+.2f}%</span></div>
                  </div>
                                    <div class="metrics">
                                        <span><b>RSI14</b> {latest_rsi:.1f} <em>{rsi_state}</em></span>
                                        <span><b>ATR14</b> ${latest_atr:.2f}</span>
                                        <span><b>SMA20</b> ${float(data['SMA20'].iloc[-1]):.2f}</span>
                                        <span><b>SMA50</b> ${float(data['SMA50'].iloc[-1]):.2f}</span>
                                    </div>
                                    <div class="signal {signal_class}">
                                        <strong>{signal['decision']}</strong>
                                        <span>{signal['confidence']:.0f}% de evidencia · {signal['buy_score']} compra / {signal['sell_score']} venta</span>
                                        <small>Triggers: {escape(trigger_text)}</small>
                                    </div>
                  {chart}
                  <table><thead><tr><th>Fecha</th><th>Pronóstico</th></tr></thead>
                  <tbody>{rows}</tbody></table>
                </article>
            """)
        except Exception as error:
            failed.append(f"{ticker}: {error}")
            print(f"[{ticker}] Dashboard error: {error}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_block = ''.join(
        f'<li>{escape(error)}</li>' for error in failed
    )
    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Analysis Dashboard</title>
<style>
* {{ box-sizing: border-box; }} body {{ margin: 0; background: #eef2f7; color: #172033;
font: 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
header {{ padding: 28px clamp(20px, 5vw, 72px); background: #172033; color: white; }}
header h1 {{ margin: 0 0 8px; font-size: clamp(24px, 4vw, 38px); }}
.meta, .muted {{ color: #718096; font-size: 12px; }} header .meta {{ color: #cbd5e1; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px clamp(16px, 4vw, 48px) 48px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 20px; }}
.card {{ background: white; border: 1px solid #dbe3ee; border-radius: 8px; padding: 18px;
box-shadow: 0 8px 20px rgba(23,32,51,.06); }}
.card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; }}
h2 {{ margin: 0 0 5px; font-size: 21px; }} .price {{ text-align: right; }}
.price strong {{ display: block; font-size: 20px; }} .positive {{ color: #15803d; }}
.negative {{ color: #b91c1c; }} table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 4px; }}
.metrics span {{ background: #f1f5f9; border-radius: 999px; padding: 6px 9px; font-size: 12px; }}
.metrics b {{ color: #334155; }} .metrics em {{ color: #64748b; font-style: normal; }}
.signal {{ display: grid; gap: 3px; border-left: 4px solid #64748b; background: #f8fafc; padding: 9px 11px; margin: 10px 0; }}
.signal strong {{ font-size: 16px; }} .signal span, .signal small {{ color: #64748b; font-size: 12px; }}
.signal.compra {{ border-color: #15803d; background: #f0fdf4; }} .signal.compra strong {{ color: #15803d; }}
.signal.venta {{ border-color: #b91c1c; background: #fef2f2; }} .signal.venta strong {{ color: #b91c1c; }}
.signal.espera {{ border-color: #ca8a04; background: #fefce8; }} .signal.espera strong {{ color: #a16207; }}
th, td {{ padding: 7px 4px; border-bottom: 1px solid #edf1f5; text-align: left; }}
th {{ color: #526174; font-size: 12px; text-transform: uppercase; }}
.notice {{ background: #fff7ed; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-bottom: 20px; }}
@media (max-width: 560px) {{ .grid {{ grid-template-columns: 1fr; }} .card {{ padding: 12px; }} }}
</style></head><body>
<header><h1>Stock Analysis Dashboard</h1>
<div class="meta">{len(cards)} activos · {days} días · {past_days} anclas históricas · generado {pd.Timestamp.now():%Y-%m-%d %H:%M}</div>
</header><main>
{f'<div class="notice"><strong>No se pudieron procesar:</strong><ul>{error_block}</ul></div>' if failed else ''}
<section class="grid">{''.join(cards)}</section>
</main></body></html>"""
    output_path.write_text(html, encoding='utf-8')
    print(f"Dashboard saved to: {output_path.resolve()}")


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
    parser.add_argument(
        "--view",
        choices=("windows", "dashboard"),
        default="windows",
        help="Visualization mode: separate windows or one HTML dashboard (default: windows)",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=Path("data/dashboard/stock_dashboard.html"),
        help="Output path for the HTML dashboard",
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
    
    # Resolve one model per ticker unless a shared model was explicitly provided.
    data_models_dir = Path("data/models/")
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"\nError: Model file not found: {model_path}")
            return
        model_paths = {ticker: model_path for ticker in args.tickers}
    else:
        model_paths = {
            ticker: find_model_for_ticker(ticker, data_models_dir)
            for ticker in args.tickers
        }
        missing_models = [ticker for ticker, path in model_paths.items() if path is None]
        if missing_models:
            print("\nError: No matching model found for: " + ", ".join(missing_models))
            print("Train the missing tickers first using train_lstm_model.py")
            return

        print("\nUsing matching model per ticker:")
        for ticker, model_path in model_paths.items():
            print(f"  {ticker}: {model_path.name}")

    if any(not model_path.exists() for model_path in model_paths.values()):
        print(f"\nError: Model file not found: {model_path}")
        return
    
    lookback = 60  # Must match training configuration
    
    if args.view == "dashboard":
        create_dashboard(
            args.tickers,
            model_paths,
            args.days,
            lookback,
            args.past_days,
            args.dashboard_output,
        )
        return

    # Create a process for each ticker
    processes = []
    
    print(f"\nLaunching {len(args.tickers)} monitoring windows...")
    print("Note: Each stock opens in a separate window.")
    print("Close all windows to exit.\n")
    
    for ticker in args.tickers:
        p = multiprocessing.Process(
            target=monitor_single_stock,
            args=(ticker, model_paths[ticker], args.days, lookback, args.past_days),
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
