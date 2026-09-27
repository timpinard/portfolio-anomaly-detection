#!/usr/bin/env python3
"""
Create a validation dashboard from existing validation results.

This script can be used to regenerate the validation dashboard
from previously saved validation data.

Usage:
    python scripts/create_validation_dashboard.py cross_sectional
    python scripts/create_validation_dashboard.py individual --timestamp 20250113_143022
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from utils.visualizations import create_validation_dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_metadata(model_dir: Path) -> dict:
    """Load training metadata."""
    metadata_path = model_dir / 'training_metadata.json'
    if not metadata_path.exists():
        logger.warning(f"Training metadata not found: {metadata_path}")
        return {}
    with open(metadata_path) as f:
        return json.load(f)


def load_price_data(db_path: Path, symbols: list, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Load price data for forward return calculation."""
    conn = sqlite3.connect(db_path)

    placeholders = ','.join('?' * len(symbols))
    query = f"""
        SELECT date, symbol, close
        FROM market_prices
        WHERE symbol IN ({placeholders})
          AND date >= ?
    """
    params = list(symbols) + [start_date]

    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    # Ensure timezone-naive for consistency
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    return df


def calculate_forward_returns(prices_df: pd.DataFrame, horizons: list = [1, 5, 20]) -> pd.DataFrame:
    """Calculate forward returns for each symbol at various horizons."""
    pivoted = prices_df.pivot(index='date', columns='symbol', values='close')

    results = None
    for horizon in horizons:
        fwd_ret = pivoted.pct_change(horizon, fill_method=None).shift(-horizon)
        fwd_ret_stacked = fwd_ret.stack().reset_index()
        fwd_ret_stacked.columns = ['date', 'symbol', f'fwd_return_{horizon}d']

        if results is None:
            results = fwd_ret_stacked
        else:
            results = results.merge(fwd_ret_stacked, on=['date', 'symbol'], how='outer')

    return results


def calculate_forward_volatility(prices_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Calculate forward realized volatility."""
    pivoted = prices_df.pivot(index='date', columns='symbol', values='close')
    daily_returns = pivoted.pct_change(fill_method=None)

    # Forward rolling volatility
    fwd_vol = daily_returns.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    fwd_vol_stacked = fwd_vol.stack().reset_index()
    fwd_vol_stacked.columns = ['date', 'symbol', f'fwd_volatility_{horizon}d']

    return fwd_vol_stacked


def create_dashboard_from_results(
    model_type: str,
    timestamp: str = None,
    forward_horizons: list = [1, 5, 20]
):
    """Create validation dashboard from existing results."""
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models' / 'model_types' / model_type
    results_dir = project_root / 'results' / 'model_types' / model_type
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.error(f"Run 'python scripts/train_model.py {model_type}' first")
        sys.exit(1)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.error(f"Run 'python scripts/validate_model.py {model_type}' first")
        sys.exit(1)
    
    # Find most recent validation results if timestamp not provided
    if timestamp is None:
        score_files = list(results_dir.glob('backtest_scores_*.csv'))
        if not score_files:
            logger.error("No validation results found. Run validation first.")
            sys.exit(1)
        # Extract timestamp from most recent file
        most_recent = max(score_files, key=lambda p: p.stat().st_mtime)
        timestamp = most_recent.stem.replace('backtest_scores_', '')
        logger.info(f"Using most recent validation results: {timestamp}")
    
    # Load model
    logger.info("Loading model...")
    ae_model = AutoencoderAnomalyDetector.load(model_dir / 'autoencoder')
    
    # Load training metadata
    training_meta = load_training_metadata(model_dir)
    
    # Load validation results
    scores_file = results_dir / f'backtest_scores_{timestamp}.csv'
    if not scores_file.exists():
        logger.error(f"Validation scores file not found: {scores_file}")
        sys.exit(1)
    
    logger.info(f"Loading validation results from: {scores_file}")
    test_df = pd.read_csv(scores_file)
    test_df['date'] = pd.to_datetime(test_df['date'])
    if test_df['date'].dt.tz is not None:
        test_df['date'] = test_df['date'].dt.tz_localize(None)
    
    # Calculate forward returns from price data
    analysis_df = None
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    
    if db_path.exists():
        logger.info("Loading price data to calculate forward returns...")
        symbols = test_df['symbol'].unique().tolist()
        test_start = test_df['date'].min().strftime('%Y-%m-%d')
        test_end = test_df['date'].max().strftime('%Y-%m-%d')
        
        try:
            prices_df = load_price_data(db_path, symbols, test_start, test_end)
            
            if not prices_df.empty:
                # Calculate forward returns
                fwd_returns = calculate_forward_returns(prices_df, forward_horizons)
                fwd_vol = calculate_forward_volatility(prices_df)
                
                # Ensure date columns are timezone-naive for merging
                if fwd_returns['date'].dt.tz is not None:
                    fwd_returns['date'] = fwd_returns['date'].dt.tz_localize(None)
                if fwd_vol['date'].dt.tz is not None:
                    fwd_vol['date'] = fwd_vol['date'].dt.tz_localize(None)
                
                # Merge with scores
                analysis_df = test_df[['date', 'symbol', 'score', 'is_anomaly']].merge(
                    fwd_returns, on=['date', 'symbol'], how='left'
                ).merge(
                    fwd_vol, on=['date', 'symbol'], how='left'
                )
                logger.info(f"  ✓ Calculated forward returns for {len(analysis_df)} samples")
            else:
                logger.warning("  ✗ No price data found")
        except Exception as e:
            logger.warning(f"  ✗ Failed to calculate forward returns: {e}")
    else:
        logger.warning(f"Database not found: {db_path}. Forward return plots will be blank.")
    
    # Create output directory
    validation_viz_dir = results_dir / 'validation_visualizations'
    validation_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dashboard
    logger.info("Generating validation dashboard...")
    output_path = validation_viz_dir / f'validation_dashboard_{timestamp}.png'
    
    try:
        create_validation_dashboard(
            test_df, analysis_df, ae_model, training_meta, forward_horizons,
            output_path=output_path
        )
        logger.info(f"✓ Dashboard saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Create validation dashboard from existing results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create dashboard from most recent validation
    python scripts/create_validation_dashboard.py cross_sectional
    
    # Create dashboard from specific validation run
    python scripts/create_validation_dashboard.py individual --timestamp 20250113_143022
        """
    )
    
    parser.add_argument(
        'model_type',
        choices=['individual', 'cross_sectional'],
        help='Model type to create dashboard for'
    )
    
    parser.add_argument(
        '--timestamp', '-t',
        type=str,
        default=None,
        help='Timestamp of validation run (default: most recent)'
    )
    
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[1, 5, 20],
        help='Forward return horizons in days (default: 1 5 20)'
    )
    
    args = parser.parse_args()
    
    create_dashboard_from_results(
        args.model_type,
        timestamp=args.timestamp,
        forward_horizons=args.horizons
    )


if __name__ == '__main__':
    main()
