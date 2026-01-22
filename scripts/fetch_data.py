"""
Fetch market universe from Yahoo Finance.

Supports two modes:
1. S&P 500 universe (default): Fetches current S&P 500 constituents from GitHub with CSV fallback
2. Custom universe: Uses predefined sector-based universe

Design goals:
- Idempotent: can be run multiple times without side effects
- Safe to re-run (gap-aware, no duplicates)
"""
import yfinance as yf
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import logging
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.universe import get_custom_universe

YEARS_BACK = 5
MARKET_PROXY = "SPY"
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "datasets/s-and-p-500-companies/main/data/constituents.csv"
)
FALLBACK_CSV_PATH = Path("data/s&p500.csv") # as of 2026-01-01

DB_PATH = Path("data/processed/market_data.sqlite")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_sp500_universe() -> tuple[pd.DataFrame, str]:
    """
    Fetch current S&P 500 securities (symbol + company name).
    Tries GitHub raw link first, falls back to local CSV file.
    Returns (df, source)
    """
    # Try GitHub raw link first, may run into proxy or bot issues
    try:
        df = pd.read_csv(GITHUB_RAW_URL)
        
        if not {"Symbol", "Security"}.issubset(df.columns):
            raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
        
        df = df[["Symbol", "Security"]].copy()
        df.columns = ["symbol", "name"]
        
        # Yahoo Finance uses '-' instead of '.'
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
        
        logger.info(f"Fetched {len(df)} S&P 500 symbols from GitHub")
        return df, "github"
        
    except Exception as e:
        logger.warning(f"GitHub source failed: {e}")
    
    # Fallback to local CSV file
    try:
        if not FALLBACK_CSV_PATH.exists():
            raise FileNotFoundError(f"Fallback CSV not found: {FALLBACK_CSV_PATH}")
        
        df = pd.read_csv(FALLBACK_CSV_PATH)
        
        if not {"Symbol", "Security"}.issubset(df.columns):
            raise ValueError(f"Unexpected columns in CSV: {df.columns.tolist()}")
        
        df = df[["Symbol", "Security"]].copy()
        df.columns = ["symbol", "name"]
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
        
        logger.info(f"Fetched {len(df)} S&P 500 symbols from local CSV")
        return df, "local_csv"
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch S&P 500 universe from all sources. "
            f"GitHub failed, and fallback CSV at {FALLBACK_CSV_PATH} also failed: {e}"
        ) from e



def recreate_tables(conn: sqlite3.Connection):
    """
    Drop and recreate tables with clean schema.
    """
    conn.execute("DROP TABLE IF EXISTS market_prices")
    
    conn.execute("""
        CREATE TABLE market_prices (
            date   TEXT,
            symbol TEXT,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume INTEGER,
            PRIMARY KEY (date, symbol)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS universe_sp500 (
            symbol TEXT PRIMARY KEY,
            name   TEXT,
            source TEXT,
            as_of  TEXT
        )
    """)
    
    conn.commit()
    logger.info("Recreated market_prices table (clean date-only schema)")


def persist_universe_snapshot(
    df: pd.DataFrame,
    source: str,
    conn: sqlite3.Connection
):
    """Persist universe snapshot to database."""
    as_of = date.today().isoformat()
    
    out = df.copy()
    out["source"] = source
    out["as_of"] = as_of
    
    conn.execute("DELETE FROM universe_sp500")
    out.to_sql("universe_sp500", conn, if_exists="append", index=False)
    conn.commit()
    
    logger.info(f"Persisted universe snapshot ({len(out)} symbols, source: {source})")

def fetch_and_store_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    conn: sqlite3.Connection
) -> int:
    """
    Fetch daily OHLCV data from Yahoo and append to market_prices.
    Enforces:
      - flat columns
      - YYYY-MM-DD dates
      - no timezone or time component
    """
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False
        )
        
        if df.empty:
            logger.warning(f"{symbol}: no data for {start_date} → {end_date}")
            return 0
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        df = df.reset_index()

        # Clean up dates and symbols
        df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
        
        df["symbol"] = symbol
        
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
        
        df.to_sql(
            "market_prices",
            conn,
            if_exists="append",
            index=False
        )
        
        return len(df)
        
    except Exception as e:
        logger.error(f"{symbol}: {e}")
        return 0

def fetch_sp500_universe_data(years_back: int = YEARS_BACK):
    """
    Fetch S&P 500 universe and hydrate prices.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    recreate_tables(conn)
    
    today = date.today()
    target_start = (today - timedelta(days=365 * years_back)).isoformat()
    today_str = today.isoformat()
    
    universe_df, universe_source = fetch_sp500_universe()
    persist_universe_snapshot(universe_df, universe_source, conn)
    
    symbols = universe_df["symbol"].tolist()
    symbols.append(MARKET_PROXY)
    
    logger.info(f"Hydrating prices for {len(symbols)} symbols")
    logger.info(f"Date range: {target_start} → {today_str}")
    
    total_rows = 0
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}")
        rows = fetch_and_store_prices(symbol, target_start, today_str, conn)
        logger.info(f"  → {rows} rows")
        total_rows += rows
    
    conn.close()
    logger.info(f"Done. Inserted {total_rows:,} rows into market_prices.")


def fetch_custom_universe_data(years_back: int = YEARS_BACK):
    """
    Fetch custom predefined universe and hydrate prices.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    recreate_tables(conn)
    
    all_symbols = get_custom_universe()
    
    logger.info(f"Fetching {len(all_symbols)} symbols for custom market universe...")
    
    today = date.today()
    target_start = (today - timedelta(days=365 * years_back)).isoformat()
    today_str = today.isoformat()
    
    logger.info(f"Date range: {target_start} → {today_str}")
    
    total_rows = 0
    
    for i, symbol in enumerate(all_symbols, 1):
        logger.info(f"[{i}/{len(all_symbols)}] {symbol}")
        rows = fetch_and_store_prices(symbol, target_start, today_str, conn)
        logger.info(f"  → {rows} rows")
        total_rows += rows
    
    conn.close()
    logger.info(f"Done. Inserted {total_rows:,} rows into market_prices.")


def fetch_market_universe():
    """
    Legacy function for backward compatibility.
    Fetches custom universe (original behavior).
    """
    fetch_custom_universe_data(years_back=10)  # Original used 10y period


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fetch market data from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch S&P 500 universe (default)
  python scripts/fetch_data.py --sp500
  
  # Fetch custom predefined universe
  python scripts/fetch_data.py --custom
  
  # Fetch S&P 500 with custom years
  python scripts/fetch_data.py --sp500 --years 10
        """
    )
    parser.add_argument(
        '--sp500',
        action='store_true',
        help='Fetch S&P 500 universe (default)'
    )
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Fetch custom predefined universe'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=YEARS_BACK,
        help=f'Number of years of historical data (default: {YEARS_BACK})'
    )
    
    args = parser.parse_args()
    
    if args.custom:
        fetch_custom_universe_data(years_back=args.years)
    else:
        fetch_sp500_universe_data(years_back=args.years)

