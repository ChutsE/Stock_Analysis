"""
Advanced script to export historical market data with customizable options.
"""

import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def save_historical_data(ticker, period="1y", interval="1d", output_dir="data", save_stats=False):
    """
    Download and save historical data with additional information.
    
    Args:
        ticker (str): Ticker symbol
        period (str): Time period
        interval (str): Data interval
        output_dir (str): Directory where files will be saved
        save_stats (bool): Whether to save the stats CSV file
    
    Returns:
        tuple: (historical filename, stats filename or None, log filename)
    """
    try:
        # Create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory created: {output_dir}")
        
        print(f"\n{'='*60}")
        print(f"Processing: {ticker}")
        print('='*60)
        
        # Download market data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        info = stock.info
        
        if hist.empty:
            print(f"No data found for {ticker}")
            return None, None
        
        # Normalize ticker for filename usage
        ticker_clean = ticker.replace('^', 'INDEX_')
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Save historical data
        hist_filename = os.path.join(output_dir, f"{ticker_clean}_historical_{period}_{date_str}.csv")
        hist.to_csv(hist_filename)
        print(f"✓ Historical data saved: {hist_filename}")
        print(f"  Records: {len(hist)}")
        print(f"  Range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
        
        if save_stats:
            # Compute additional statistics
            hist_with_stats = hist.copy()
            hist_with_stats['Daily_Return'] = hist_with_stats['Close'].pct_change() * 100
            hist_with_stats['Cumulative_Return'] = ((hist_with_stats['Close'] / hist_with_stats['Close'].iloc[0]) - 1) * 100
            hist_with_stats['MA_20'] = hist_with_stats['Close'].rolling(window=20).mean()
            hist_with_stats['MA_50'] = hist_with_stats['Close'].rolling(window=50).mean()
            hist_with_stats['Average_OHLC'] = (
                hist_with_stats['Open']
                + hist_with_stats['High']
                + hist_with_stats['Low']
                + hist_with_stats['Close']
            ) / 4
            hist_with_stats['Average_Close_To_Date'] = hist_with_stats['Close'].expanding().mean()
            hist_with_stats['Max_20'] = hist_with_stats['High'].rolling(window=20).max()
            hist_with_stats['Min_20'] = hist_with_stats['Low'].rolling(window=20).min()
            hist_with_stats['Max_50'] = hist_with_stats['High'].rolling(window=50).max()
            hist_with_stats['Min_50'] = hist_with_stats['Low'].rolling(window=50).min()
            hist_with_stats['All_Time_Max'] = hist_with_stats['High'].expanding().max()
            hist_with_stats['All_Time_Min'] = hist_with_stats['Low'].expanding().min()

            # Save stats data
            stats_filename = os.path.join(output_dir, f"{ticker_clean}_with_stats_{period}_{date_str}.csv")
            hist_with_stats.to_csv(stats_filename)
            print(f"✓ Stats data saved: {stats_filename}")
        else:
            stats_filename = None
            print(f"- Skipped with_stats file for {ticker}")
        
        # Build information summary
        summary = {
            'Ticker': ticker,
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Exchange': info.get('exchange', 'N/A'),
            'Current Price': info.get('regularMarketPrice', 'N/A'),
            'Previous Close': info.get('previousClose', 'N/A'),
            'Volume': info.get('volume', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Data Period': period,
            'Total Records': len(hist),
            'Download Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary to log (instead of CSV)
        log_filename = os.path.join(output_dir, f"{ticker_clean}_info_{date_str}.log")
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write(f"Information summary - {ticker}\n")
            log_file.write("=" * 60 + "\n")
            for key, value in summary.items():
                log_file.write(f"{key}: {value}\n")
        print(f"✓ Information summary saved to log: {log_filename}")

        print("\nSummary (log):")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Print period statistics
        print(f"\nPeriod statistics:")
        print(f"  Starting Price: ${hist['Close'].iloc[0]:.2f}")
        print(f"  Ending Price: ${hist['Close'].iloc[-1]:.2f}")
        print(f"  Total Return: {((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100:.2f}%")
        print(f"  Max Price: ${hist['High'].max():.2f}")
        print(f"  Min Price: ${hist['Low'].min():.2f}")
        print(f"  Average Price (Close): ${hist['Close'].mean():.2f}")
        print(f"  Average Volume: {hist['Volume'].mean():,.0f}")
        
        return hist_filename, stats_filename, log_filename
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Advanced market historical data exporter.\n\n"
            "Supported instrument/ticker types:\n"
            "- Stocks: AAPL, MSFT, TSLA\n"
            "- ETFs: IVVPESO.MX, IVV, SPY, QQQ\n"
            "- Indexes: ^GSPC, ^DJI, ^IXIC\n"
            "- Forex: EURUSD=X\n"
            "- Crypto: BTC-USD, ETH-USD\n\n"
            "Where to find tickers:\n"
            "- Yahoo Finance: https://finance.yahoo.com\n"
            "- TradingView symbols search: https://www.tradingview.com/symbols/\n"
            "- Mexican market (BMV/SIC) references: https://www.bmv.com.mx\n\n"
            "Tips:\n"
            "- S&P 500 index in Yahoo: ^GSPC\n"
            "- Mexico-listed symbols in Yahoo usually use .MX (example: IVVPESO.MX)"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        help="One or more tickers. Example: AAPL IVV ^GSPC",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Data period (default: 1y). Ex: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval (default: 1d). Ex: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Generate with_stats file (disabled by default)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("ADVANCED HISTORICAL DATA EXPORTER")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Period: {args.period}")
    print(f"  Interval: {args.interval}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Save stats: {args.stats}")
    
    # Process each ticker
    archivos_creados = []
    
    for ticker in args.tickers:
        hist_file, stats_file, log_file = save_historical_data(
            ticker=ticker,
            period=args.period,
            interval=args.interval,
            output_dir=args.output_dir,
            save_stats=args.stats,
        )
        
        if hist_file:
            archivos_creados.append(hist_file)
            if stats_file:
                archivos_creados.append(stats_file)
            if log_file:
                archivos_creados.append(log_file)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PROCESS COMPLETED")
    print("=" * 70)
    print(f"Total de archivos creados: {len(archivos_creados)}")
    print(f"\nFiles saved in: {os.path.abspath(args.output_dir)}")
    for archivo in archivos_creados:
        if archivo:
            print(f"  - {os.path.basename(archivo)}")
    print("\nExport completed successfully!")


if __name__ == "__main__":
    main()
