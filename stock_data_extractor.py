"""
Stock Data Extractor for S&P 500 and IVV
Extracts live stock data using yfinance library
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def get_live_stock_data(ticker):
    """
    Get live stock data for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary with stock information
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get current price and info
        info = stock.info
        
        # Get the most recent data
        hist = stock.history(period="1d", interval="1m")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            current_time = hist.index[-1]
        else:
            current_price = info.get('currentPrice', 'N/A')
            current_time = datetime.now()
        
        data = {
            'Ticker': ticker,
            'Current Price': current_price,
            'Time': current_time,
            'Previous Close': info.get('previousClose', 'N/A'),
            'Open': info.get('open', 'N/A'),
            'Day High': info.get('dayHigh', 'N/A'),
            'Day Low': info.get('dayLow', 'N/A'),
            'Volume': info.get('volume', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
        }
        
        return data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


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


def main():
    """Main function to extract stock data"""
    
    # Define tickers
    # ^GSPC is the S&P 500 index
    # SPY is the S&P 500 ETF
    # IVV is the iShares Core S&P 500 ETF
    tickers = ['^GSPC', 'SPY', 'IVV']
    
    print("=" * 80)
    print("LIVE STOCK DATA EXTRACTION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Get live data for each ticker
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Data for {ticker}")
        print('='*60)
        
        data = get_live_stock_data(ticker)
        
        if data:
            for key, value in data.items():
                if key != 'Ticker':
                    print(f"{key:20s}: {value}")
        print()
    
    # Get historical data example
    print("\n" + "="*80)
    print("HISTORICAL DATA (Last 5 days)")
    print("="*80)
    
    for ticker in ['SPY', 'IVV']:
        print(f"\n{ticker} - Last 5 days:")
        hist = get_historical_data(ticker, period="5d", interval="1d")
        if hist is not None and not hist.empty:
            print(hist[['Open', 'High', 'Low', 'Close', 'Volume']])
        print()


if __name__ == "__main__":
    main()
