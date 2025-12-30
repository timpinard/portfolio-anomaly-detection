"""Fetch market universe from Yahoo Finance."""
import yfinance as yf
import sqlite3
from pathlib import Path

def fetch_market_universe():
    """Fetch proper market universe - FREE, no limits."""
    
    # Comprehensive universe
    universe = {
        # Market indices
        'indices': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
        
        # Tech
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ORCL', 'CSCO', 'INTC'],
        
        # Finance  
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'C', 'MS', 'SCHW', 'BLK'],
        
        # Healthcare
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'ABT'],
        
        # Consumer
        'consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT'],
        
        # Energy
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO'],
        
        # Industrials
        'industrials': ['BA', 'CAT', 'HON', 'UPS', 'GE', 'MMM', 'LMT', 'RTX']
    }
    
    all_symbols = []
    for sector, symbols in universe.items():
        all_symbols.extend(symbols)
    
    print(f"Fetching {len(all_symbols)} symbols for market universe...")
    
    db_path = Path("data/processed/market_data.sqlite")
    conn = sqlite3.connect(db_path)
    
    # Clear existing data
    conn.execute("DELETE FROM market_prices")
    conn.commit()
    
    for i, symbol in enumerate(all_symbols, 1):
        print(f"[{i}/{len(all_symbols)}] {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5y")
            
            df = df.reset_index()
            df['symbol'] = symbol
            df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                'market_prices', conn, if_exists='append', index=False
            )
            print(f"  ✓ {len(df)} records")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    conn.close()
    print("Done!")

if __name__ == '__main__':
    fetch_market_universe()

