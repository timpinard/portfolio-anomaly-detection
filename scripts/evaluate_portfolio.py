#!/usr/bin/env python3
"""
Portfolio evaluation script.

Evaluates a portfolio of stocks against trained anomaly detection models.
Also evaluates the current scores for each security in the original training set
and provides weighted aggregate scores for both the training set basket and the portfolio.
"""

import sys
import argparse
from pathlib import Path
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector
from data.feature_extractor import MarketFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_from_yahoo_and_save(symbol: str, db_path: Path) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance and save to database.
    
    Args:
        symbol: Ticker symbol
        db_path: Path to SQLite database
    
    Returns:
        DataFrame with price data
    """
    logger.info(f"Fetching {symbol} from Yahoo Finance...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3y")
        
        if df.empty:
            logger.warning(f"No data returned from Yahoo Finance for {symbol}")
            return pd.DataFrame()
        
        # Reset index and format
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
        
        # Ensure date is string format
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.strftime('%Y-%m-%d')
        
        # Save to database
        conn = sqlite3.connect(db_path)
        
        # Create table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        """)
        
        # Delete existing data for this symbol (replace)
        conn.execute("DELETE FROM market_prices WHERE symbol = ?", (symbol,))
        
        # Insert new data
        df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].to_sql(
            'market_prices', conn, if_exists='append', index=False
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Fetched and saved {len(df)} records for {symbol}")
        
        # Format for return (set date as index)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} from Yahoo Finance: {e}")
        return pd.DataFrame()


def ensure_symbol_data(symbol: str, db_path: Path, extractor: MarketFeatureExtractor) -> pd.DataFrame:
    """
    Ensure symbol data exists in database, fetching from Yahoo Finance if needed.
    
    Args:
        symbol: Ticker symbol
        db_path: Path to SQLite database
        extractor: MarketFeatureExtractor instance
    
    Returns:
        DataFrame with price data
    """
    # Try to load from database
    df = extractor.load_price_data(symbol)
    
    # If empty or insufficient data, fetch from Yahoo Finance
    if df.empty or len(df) < 60:  # Require at least 60 days for feature calculation
        logger.info(f"Symbol {symbol} not found in database or insufficient data, fetching from Yahoo Finance...")
        df = fetch_from_yahoo_and_save(symbol, db_path)
        
        # Reload from database after saving
        if not df.empty:
            df = extractor.load_price_data(symbol)
    
    return df


def get_training_set_symbols(db_path: Path) -> List[str]:
    """
    Get list of symbols from the training set (market_features table).
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        List of unique symbols
    """
    conn = sqlite3.connect(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_features ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not read market_features table: {e}")
        logger.info("Trying to get symbols from market_prices table instead...")
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return symbols


def score_security(
    symbol: str,
    extractor: MarketFeatureExtractor,
    db_path: Path,
    ae_model: AutoencoderAnomalyDetector,
    if_model: IsolationForestAnomalyDetector,
    backfill: bool = True,
    shares: Optional[float] = None
) -> Optional[Dict]:
    """
    Score a single security.
    
    Args:
        symbol: Ticker symbol
        extractor: MarketFeatureExtractor instance
        db_path: Path to SQLite database
        ae_model: Autoencoder model
        if_model: Isolation Forest model
        backfill: Whether to backfill missing data
        shares: Number of shares owned (optional)
    
    Returns:
        Dictionary with scores and metadata, or None if failed
    """
    try:
        # Ensure data exists
        if backfill:
            df = ensure_symbol_data(symbol, db_path, extractor)
        else:
            df = extractor.load_price_data(symbol)
        
        if df.empty or len(df) < 60:
            logger.warning(f"Insufficient data for {symbol} (need at least 60 days)")
            return None
        
        # Calculate features
        features = extractor.calculate_all_features(df)
        if features.empty:
            logger.warning(f"Could not calculate features for {symbol}")
            return None
        
        # Get most recent features
        latest_features = features.iloc[-1]
        
        # Prepare feature matrix (exclude symbol column)
        feature_cols = extractor.get_feature_columns(features)
        feature_values = [latest_features[col] for col in feature_cols]
        X = np.array([feature_values])
        
        # Replace inf and NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Score with both models
        ae_score = float(ae_model.score(X)[0])
        ae_prediction = int(ae_model.predict(X)[0])
        ae_is_anomaly = (ae_prediction == -1)
        
        if_score = float(if_model.score(X)[0])
        if_prediction = int(if_model.predict(X)[0])
        if_is_anomaly = (if_prediction == -1)
        
        # Get current price for weighting
        current_price = float(df['close'].iloc[-1])
        
        result = {
            'symbol': symbol,
            'ae_score': ae_score,
            'ae_threshold': float(ae_model.threshold),
            'ae_is_anomaly': ae_is_anomaly,
            'if_score': if_score,
            'if_is_anomaly': if_is_anomaly,
            'consensus_anomaly': ae_is_anomaly and if_is_anomaly,
            'current_price': current_price,
            'date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
        }
        
        # Add shares if provided
        if shares is not None:
            result['shares'] = shares
            result['position_value'] = shares * current_price
        
        return result
        
    except Exception as e:
        logger.error(f"Error scoring {symbol}: {e}")
        return None


def zscore_against_basket(
    scores: List[Dict],
    basket_scores: List[Dict],
    score_key: str = 'if_score'
) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    """
    Compute z-scores for scores relative to a reference basket.
    
    Args:
        scores: List of score dictionaries to compute z-scores for
        basket_scores: List of score dictionaries from the reference basket
        score_key: Key in score dict to use for z-score calculation (default: 'if_score')
    
    Returns:
        Tuple of (basket_mean, basket_std, updated_scores)
        Returns (None, None, scores) if insufficient data
    """
    # Extract basket scores
    basket_values = [
        s[score_key] for s in basket_scores
        if s.get(score_key) is not None
    ]
    
    if len(basket_values) < 2:
        logger.warning(f"Insufficient basket data for z-score calculation ({len(basket_values)} scores)")
        return None, None, scores
    
    mean = np.mean(basket_values)
    std = np.std(basket_values)
    
    if std == 0:
        std = 1e-8  # Avoid divide-by-zero
    
    # Add z-scores to all scores
    z_score_key = f'{score_key}_z_score'
    for s in scores:
        if s.get(score_key) is not None:
            s[z_score_key] = (s[score_key] - mean) / std
        else:
            s[z_score_key] = None
    
    return mean, std, scores


def calculate_weights_from_shares(scores: List[Dict], shares: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Calculate portfolio weights from shares and current prices.
    
    Args:
        scores: List of score dictionaries (must include current_price)
        shares: Optional dictionary of shares by symbol (if None, uses equal weights)
    
    Returns:
        Dictionary of weights by symbol
    """
    if not scores:
        return {}
    
    # If shares provided, calculate weights from position values
    if shares is not None:
        total_value = sum(shares.get(s['symbol'], 0) * s['current_price'] for s in scores)
        if total_value == 0:
            # Fallback to equal weights
            n = len(scores)
            return {s['symbol']: 1.0 / n for s in scores}
        
        weights = {}
        for s in scores:
            position_value = shares.get(s['symbol'], 0) * s['current_price']
            weights[s['symbol']] = position_value / total_value
        return weights
    
    # Equal weights if no shares provided
    n = len(scores)
    return {s['symbol']: 1.0 / n for s in scores}


def calculate_weighted_score(scores: List[Dict], weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Calculate weighted aggregate scores.
    
    Args:
        scores: List of score dictionaries
        weights: Optional dictionary of weights by symbol (if None, uses equal weights)
    
    Returns:
        Dictionary with weighted aggregate scores
    """
    if not scores:
        return {
            'ae_weighted_score': 0.0,
            'if_weighted_score': 0.0,
            'n_securities': 0
        }
    
    # Calculate weights if not provided (equal weights)
    if weights is None:
        n = len(scores)
        weights = {s['symbol']: 1.0 / n for s in scores}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.get(s['symbol'], 0) for s in scores)
    if total_weight == 0:
        # Fallback to equal weights
        n = len(scores)
        weights = {s['symbol']: 1.0 / n for s in scores}
        total_weight = 1.0
    
    normalized_weights = {s['symbol']: weights.get(s['symbol'], 0) / total_weight for s in scores}
    
    # Calculate weighted scores
    ae_weighted = sum(s['ae_score'] * normalized_weights.get(s['symbol'], 0) for s in scores)
    if_weighted = sum(s['if_score'] * normalized_weights.get(s['symbol'], 0) for s in scores)
    
    # Calculate weighted z-scores if available
    ae_z_weighted = None
    if_z_weighted = None
    if any(s.get('ae_score_z_score') is not None for s in scores):
        ae_z_weighted = sum(
            s.get('ae_score_z_score', 0) * normalized_weights.get(s['symbol'], 0)
            for s in scores
            if s.get('ae_score_z_score') is not None
        )
    if any(s.get('if_score_z_score') is not None for s in scores):
        if_z_weighted = sum(
            s.get('if_score_z_score', 0) * normalized_weights.get(s['symbol'], 0)
            for s in scores
            if s.get('if_score_z_score') is not None
        )
    
    # Count anomalies
    ae_anomalies = sum(1 for s in scores if s['ae_is_anomaly'])
    if_anomalies = sum(1 for s in scores if s['if_is_anomaly'])
    consensus_anomalies = sum(1 for s in scores if s['consensus_anomaly'])
    
    result = {
        'ae_weighted_score': ae_weighted,
        'if_weighted_score': if_weighted,
        'ae_anomaly_count': ae_anomalies,
        'if_anomaly_count': if_anomalies,
        'consensus_anomaly_count': consensus_anomalies,
        'n_securities': len(scores),
        'anomaly_rate_ae': ae_anomalies / len(scores) if scores else 0.0,
        'anomaly_rate_if': if_anomalies / len(scores) if scores else 0.0,
        'anomaly_rate_consensus': consensus_anomalies / len(scores) if scores else 0.0,
        'weights': normalized_weights
    }
    
    # Add z-score aggregates if available
    if ae_z_weighted is not None:
        result['ae_weighted_z_score'] = ae_z_weighted
    if if_z_weighted is not None:
        result['if_weighted_z_score'] = if_z_weighted
    
    return result


def evaluate_training_set(
    extractor: MarketFeatureExtractor,
    db_path: Path,
    ae_model: AutoencoderAnomalyDetector,
    if_model: IsolationForestAnomalyDetector
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate all securities in the training set.
    
    Returns:
        Tuple of (individual scores list, weighted aggregate dict)
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATING TRAINING SET")
    logger.info("="*80)
    
    # Get training set symbols
    symbols = get_training_set_symbols(db_path)
    logger.info(f"Found {len(symbols)} symbols in training set")
    
    # Score each symbol
    scores = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Scoring {symbol}...")
        score = score_security(symbol, extractor, db_path, ae_model, if_model, backfill=False)
        if score:
            scores.append(score)
        else:
            logger.warning(f"  Failed to score {symbol}")
    
    logger.info(f"\nSuccessfully scored {len(scores)}/{len(symbols)} securities")
    
    # Compute z-scores relative to the training set basket (for both AE and IF)
    if scores:
        # Compute z-scores for Isolation Forest scores (relative to itself)
        if_mean, if_std, scores = zscore_against_basket(scores, scores, score_key='if_score')
        # Compute z-scores for Autoencoder scores (relative to itself)
        ae_mean, ae_std, scores = zscore_against_basket(scores, scores, score_key='ae_score')
        
        if if_mean is not None and if_std is not None:
            logger.info(f"Basket IF score stats: mean={if_mean:.6f}, std={if_std:.6f}")
        if ae_mean is not None and ae_std is not None:
            logger.info(f"Basket AE score stats: mean={ae_mean:.6f}, std={ae_std:.6f}")
    
    # Calculate weighted aggregate (equal weights for training set)
    aggregate = calculate_weighted_score(scores)
    
    # Add basket statistics to aggregate
    if scores:
        if_mean = np.mean([s['if_score'] for s in scores if s.get('if_score') is not None])
        if_std = np.std([s['if_score'] for s in scores if s.get('if_score') is not None])
        ae_mean = np.mean([s['ae_score'] for s in scores if s.get('ae_score') is not None])
        ae_std = np.std([s['ae_score'] for s in scores if s.get('ae_score') is not None])
        aggregate['basket_if_mean'] = if_mean
        aggregate['basket_if_std'] = if_std
        aggregate['basket_ae_mean'] = ae_mean
        aggregate['basket_ae_std'] = ae_std
    
    return scores, aggregate


def evaluate_portfolio(
    portfolio_symbols: List[str],
    portfolio_shares: Optional[Dict[str, float]],
    extractor: MarketFeatureExtractor,
    db_path: Path,
    ae_model: AutoencoderAnomalyDetector,
    if_model: IsolationForestAnomalyDetector,
    training_scores: Optional[List[Dict]] = None
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate portfolio securities.
    
    Args:
        portfolio_symbols: List of ticker symbols in portfolio
        portfolio_shares: Optional dictionary of shares by symbol
        extractor: MarketFeatureExtractor instance
        db_path: Path to SQLite database
        ae_model: Autoencoder model
        if_model: Isolation Forest model
    
    Returns:
        Tuple of (individual scores list, weighted aggregate dict)
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATING PORTFOLIO")
    logger.info("="*80)
    logger.info(f"Portfolio symbols: {', '.join(portfolio_symbols)}")
    if portfolio_shares:
        logger.info(f"Portfolio shares: {portfolio_shares}")
    
    # Score each portfolio symbol
    scores = []
    for i, symbol in enumerate(portfolio_symbols, 1):
        logger.info(f"[{i}/{len(portfolio_symbols)}] Scoring {symbol}...")
        shares = portfolio_shares.get(symbol) if portfolio_shares else None
        score = score_security(symbol, extractor, db_path, ae_model, if_model, backfill=True, shares=shares)
        if score:
            scores.append(score)
        else:
            logger.warning(f"  Failed to score {symbol}")
    
    logger.info(f"\nSuccessfully scored {len(scores)}/{len(portfolio_symbols)} securities")
    
    # Compute z-scores relative to training set basket if available
    if training_scores:
        # Compute z-scores for Isolation Forest scores
        if_mean, if_std, scores = zscore_against_basket(scores, training_scores, score_key='if_score')
        # Compute z-scores for Autoencoder scores
        ae_mean, ae_std, scores = zscore_against_basket(scores, training_scores, score_key='ae_score')
        
        if if_mean is not None and if_std is not None:
            logger.info(f"Portfolio z-scores computed relative to basket (IF: mean={if_mean:.6f}, std={if_std:.6f})")
        if ae_mean is not None and ae_std is not None:
            logger.info(f"Portfolio z-scores computed relative to basket (AE: mean={ae_mean:.6f}, std={ae_std:.6f})")
    else:
        logger.info("Training set scores not available, skipping z-score calculation")
    
    # Calculate weights from shares and prices
    weights = calculate_weights_from_shares(scores, portfolio_shares)
    logger.info(f"Calculated portfolio weights: {weights}")
    
    # Calculate weighted aggregate
    aggregate = calculate_weighted_score(scores, weights)
    
    return scores, aggregate


def print_results(
    training_scores: List[Dict],
    training_aggregate: Dict,
    portfolio_scores: List[Dict],
    portfolio_aggregate: Dict,
    ae_model: AutoencoderAnomalyDetector
):
    """Print evaluation results."""
    print("\n" + "="*100)
    print("PORTFOLIO EVALUATION RESULTS")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Training Set Summary
    print("TRAINING SET BASKET SUMMARY")
    print("-"*100)
    print(f"Number of securities: {training_aggregate['n_securities']}")
    print(f"Autoencoder weighted score: {training_aggregate['ae_weighted_score']:.6f}")
    if 'basket_ae_mean' in training_aggregate:
        print(f"Basket AE score stats: mean={training_aggregate['basket_ae_mean']:.6f}, std={training_aggregate['basket_ae_std']:.6f}")
    print(f"Autoencoder threshold: {ae_model.threshold:.6f}")
    print(f"Isolation Forest weighted score: {training_aggregate['if_weighted_score']:.6f}")
    if 'basket_if_mean' in training_aggregate:
        print(f"Basket IF score stats: mean={training_aggregate['basket_if_mean']:.6f}, std={training_aggregate['basket_if_std']:.6f}")
    print(f"Anomaly detection rate (AE): {training_aggregate['anomaly_rate_ae']*100:.2f}% ({training_aggregate['ae_anomaly_count']} securities)")
    print(f"Anomaly detection rate (IF): {training_aggregate['anomaly_rate_if']*100:.2f}% ({training_aggregate['if_anomaly_count']} securities)")
    print(f"Consensus anomaly rate: {training_aggregate['anomaly_rate_consensus']*100:.2f}% ({training_aggregate['consensus_anomaly_count']} securities)")
    print()
    
    # Portfolio Summary
    print("PORTFOLIO SUMMARY")
    print("-"*100)
    print(f"Number of securities: {portfolio_aggregate['n_securities']}")
    print(f"Autoencoder weighted score: {portfolio_aggregate['ae_weighted_score']:.6f}")
    if 'ae_weighted_z_score' in portfolio_aggregate and portfolio_aggregate['ae_weighted_z_score'] is not None:
        print(f"Autoencoder weighted z-score: {portfolio_aggregate['ae_weighted_z_score']:+.3f} (vs basket)")
    print(f"Isolation Forest weighted score: {portfolio_aggregate['if_weighted_score']:.6f}")
    if 'if_weighted_z_score' in portfolio_aggregate and portfolio_aggregate['if_weighted_z_score'] is not None:
        print(f"Isolation Forest weighted z-score: {portfolio_aggregate['if_weighted_z_score']:+.3f} (vs basket)")
    print(f"Anomaly detection rate (AE): {portfolio_aggregate['anomaly_rate_ae']*100:.2f}% ({portfolio_aggregate['ae_anomaly_count']} securities)")
    print(f"Anomaly detection rate (IF): {portfolio_aggregate['anomaly_rate_if']*100:.2f}% ({portfolio_aggregate['if_anomaly_count']} securities)")
    print(f"Consensus anomaly rate: {portfolio_aggregate['anomaly_rate_consensus']*100:.2f}% ({portfolio_aggregate['consensus_anomaly_count']} securities)")
    print()
    
    # Comparison
    print("COMPARISON")
    print("-"*100)
    ae_diff = portfolio_aggregate['ae_weighted_score'] - training_aggregate['ae_weighted_score']
    if_diff = portfolio_aggregate['if_weighted_score'] - training_aggregate['if_weighted_score']
    
    print(f"Portfolio AE score vs Training Set: {ae_diff:+.6f}")
    if ae_diff > ae_model.threshold * 0.1:
        print("  ⚠️  Portfolio has significantly higher anomaly score than training set")
    elif ae_diff < -ae_model.threshold * 0.1:
        print("  ✓ Portfolio has lower anomaly score than training set")
    else:
        print("  → Portfolio score is similar to training set")
    
    print(f"Portfolio IF score vs Training Set: {if_diff:+.6f}")
    
    # Z-score interpretation
    if 'if_weighted_z_score' in portfolio_aggregate and portfolio_aggregate['if_weighted_z_score'] is not None:
        if_z = portfolio_aggregate['if_weighted_z_score']
        print(f"\nPortfolio IF z-score: {if_z:+.3f}")
        if if_z > 2:
            print("  Portfolio is much more anomalous than the market today (>2σ)")
        elif if_z > 1:
            print("  Portfolio is more anomalous than the market today (>1σ)")
        elif if_z < -1:
            print("  ✓ Portfolio is calmer/more normal than the market today (<-1σ)")
        else:
            print("  → Portfolio behaves similarly to the market today")
    
    if 'ae_weighted_z_score' in portfolio_aggregate and portfolio_aggregate['ae_weighted_z_score'] is not None:
        ae_z = portfolio_aggregate['ae_weighted_z_score']
        print(f"Portfolio AE z-score: {ae_z:+.3f}")
        if ae_z > 2:
            print("  Portfolio is much more anomalous than the market today (>2σ)")
        elif ae_z > 1:
            print("  Portfolio is more anomalous than the market today (>1σ)")
        elif ae_z < -1:
            print("  ✓ Portfolio is calmer/more normal than the market today (<-1σ)")
        else:
            print("  → Portfolio behaves similarly to the market today")
    
    print()
    
    # Portfolio Details
    print("PORTFOLIO SECURITIES DETAIL")
    print("-"*100)
    has_shares = any('shares' in s for s in portfolio_scores)
    has_z_scores = any(s.get('if_score_z_score') is not None for s in portfolio_scores)
    
    if has_shares:
        if has_z_scores:
            print(f"{'Symbol':<10} {'Shares':<12} {'Weight':<10} {'Value':<12} {'IF Score':<12} {'IF Z':<8} {'IF Anom':<9} {'AE Score':<12} {'AE Z':<8} {'AE Anom':<9} {'Price':<12} {'Date':<12}")
        else:
            print(f"{'Symbol':<10} {'Shares':<12} {'Weight':<10} {'Value':<12} {'AE Score':<12} {'AE Anomaly':<12} {'IF Score':<12} {'IF Anomaly':<12} {'Price':<12} {'Date':<12}")
        print("-"*100)
        weights = portfolio_aggregate.get('weights', {})
        for score in sorted(portfolio_scores, key=lambda x: x.get('position_value', 0) or x['ae_score'], reverse=True):
            ae_anom = "✓" if score['ae_is_anomaly'] else " "
            if_anom = "✓" if score['if_is_anomaly'] else " "
            shares = score.get('shares', 0)
            weight = weights.get(score['symbol'], 0) * 100
            value = score.get('position_value', 0)
            if has_z_scores:
                if_z = score.get('if_score_z_score', None)
                ae_z = score.get('ae_score_z_score', None)
                if_z_str = f"{if_z:+.2f}" if if_z is not None else "N/A"
                ae_z_str = f"{ae_z:+.2f}" if ae_z is not None else "N/A"
                print(f"{score['symbol']:<10} {shares:<12.2f} {weight:<10.2f}% ${value:<11.2f} {score['if_score']:<12.6f} {if_z_str:<8} {if_anom:<9} {score['ae_score']:<12.6f} {ae_z_str:<8} {ae_anom:<9} ${score['current_price']:<11.2f} {score['date']:<12}")
            else:
                print(f"{score['symbol']:<10} {shares:<12.2f} {weight:<10.2f}% ${value:<11.2f} {score['ae_score']:<12.6f} {ae_anom:<12} {score['if_score']:<12.6f} {if_anom:<12} ${score['current_price']:<11.2f} {score['date']:<12}")
    else:
        if has_z_scores:
            print(f"{'Symbol':<10} {'Weight':<10} {'IF Score':<12} {'IF Z':<8} {'IF Anom':<9} {'AE Score':<12} {'AE Z':<8} {'AE Anom':<9} {'Price':<12} {'Date':<12}")
        else:
            print(f"{'Symbol':<10} {'Weight':<10} {'AE Score':<12} {'AE Anomaly':<12} {'IF Score':<12} {'IF Anomaly':<12} {'Price':<12} {'Date':<12}")
        print("-"*100)
        weights = portfolio_aggregate.get('weights', {})
        for score in sorted(portfolio_scores, key=lambda x: x['ae_score'], reverse=True):
            ae_anom = "✓" if score['ae_is_anomaly'] else " "
            if_anom = "✓" if score['if_is_anomaly'] else " "
            weight = weights.get(score['symbol'], 0) * 100
            if has_z_scores:
                if_z = score.get('if_score_z_score', None)
                ae_z = score.get('ae_score_z_score', None)
                if_z_str = f"{if_z:+.2f}" if if_z is not None else "N/A"
                ae_z_str = f"{ae_z:+.2f}" if ae_z is not None else "N/A"
                print(f"{score['symbol']:<10} {weight:<10.2f}% {score['if_score']:<12.6f} {if_z_str:<8} {if_anom:<9} {score['ae_score']:<12.6f} {ae_z_str:<8} {ae_anom:<9} ${score['current_price']:<11.2f} {score['date']:<12}")
            else:
                print(f"{score['symbol']:<10} {weight:<10.2f}% {score['ae_score']:<12.6f} {ae_anom:<12} {score['if_score']:<12.6f} {if_anom:<12} ${score['current_price']:<11.2f} {score['date']:<12}")
    
    print()
    
    # Top anomalies in training set
    print("TOP ANOMALIES IN TRAINING SET (Top 10 by AE Score)")
    print("-"*100)
    top_anomalies = sorted(training_scores, key=lambda x: x['ae_score'], reverse=True)[:10]
    has_z_scores = any(s.get('if_score_z_score') is not None for s in top_anomalies)
    
    if has_z_scores:
        print(f"{'Symbol':<10} {'IF Score':<12} {'IF Z':<8} {'IF Anom':<9} {'AE Score':<12} {'AE Z':<8} {'AE Anom':<9} {'Price':<12} {'Date':<12}")
    else:
        print(f"{'Symbol':<10} {'AE Score':<12} {'AE Anomaly':<12} {'IF Score':<12} {'IF Anomaly':<12} {'Price':<12} {'Date':<12}")
    print("-"*100)
    
    for score in top_anomalies:
        ae_anom = "✓" if score['ae_is_anomaly'] else " "
        if_anom = "✓" if score['if_is_anomaly'] else " "
        if has_z_scores:
            if_z = score.get('if_score_z_score', None)
            ae_z = score.get('ae_score_z_score', None)
            if_z_str = f"{if_z:+.2f}" if if_z is not None else "N/A"
            ae_z_str = f"{ae_z:+.2f}" if ae_z is not None else "N/A"
            print(f"{score['symbol']:<10} {score['if_score']:<12.6f} {if_z_str:<8} {if_anom:<9} {score['ae_score']:<12.6f} {ae_z_str:<8} {ae_anom:<9} ${score['current_price']:<11.2f} {score['date']:<12}")
        else:
            print(f"{score['symbol']:<10} {score['ae_score']:<12.6f} {ae_anom:<12} {score['if_score']:<12.6f} {if_anom:<12} ${score['current_price']:<11.2f} {score['date']:<12}")
    
    print()
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate portfolio against anomaly detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a portfolio with equal weights (no shares specified)
  python scripts/evaluate_portfolio.py AAPL MSFT GOOGL
  
  # Evaluate with shares (weights calculated from shares * prices)
  python scripts/evaluate_portfolio.py AAPL MSFT GOOGL --shares 100,50,25
  
  # Alternative format: symbol:shares pairs
  python scripts/evaluate_portfolio.py AAPL:100 MSFT:50 GOOGL:25
        """
    )
    
    parser.add_argument(
        'symbols',
        nargs='+',
        help='Ticker symbols in the portfolio (or symbol:shares pairs like AAPL:100)'
    )
    
    parser.add_argument(
        '--shares',
        type=str,
        help='Comma-separated number of shares for each symbol (must match order of symbols)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip evaluation of training set (faster)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (optional)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols and shares
    portfolio_symbols = []
    portfolio_shares = {}
    
    # Check if symbols are in format "SYMBOL:SHARES"
    for item in args.symbols:
        if ':' in item:
            parts = item.split(':')
            if len(parts) != 2:
                logger.error(f"Invalid format: {item}. Use SYMBOL:SHARES (e.g., AAPL:100)")
                sys.exit(1)
            symbol = parts[0].strip().upper()
            try:
                shares = float(parts[1].strip())
                portfolio_symbols.append(symbol)
                portfolio_shares[symbol] = shares
            except ValueError:
                logger.error(f"Invalid shares value: {parts[1]}")
                sys.exit(1)
        else:
            portfolio_symbols.append(item.strip().upper())
    
    # Parse shares from --shares argument if provided
    if args.shares:
        try:
            share_values = [float(s.strip()) for s in args.shares.split(',')]
            if len(share_values) != len(portfolio_symbols):
                logger.error(f"Number of shares ({len(share_values)}) must match number of symbols ({len(portfolio_symbols)})")
                sys.exit(1)
            
            for symbol, shares in zip(portfolio_symbols, share_values):
                portfolio_shares[symbol] = shares
        except ValueError as e:
            logger.error(f"Invalid shares format: {e}")
            sys.exit(1)
    
    if portfolio_shares:
        logger.info(f"Portfolio shares: {portfolio_shares}")
    else:
        logger.info("No shares specified, using equal weights")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'market_universe'
    
    # Check database exists
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run 'python scripts/fetch_data.py' first")
        sys.exit(1)
    
    # Check models exist
    ae_path = model_dir / 'autoencoder' / 'model.pth'
    if_path = model_dir / 'isolation_forest' / 'model.joblib'
    
    if not ae_path.exists() or not if_path.exists():
        logger.error(f"Models not found: {model_dir}")
        logger.error("Run 'python scripts/train_models.py' first")
        sys.exit(1)
    
    # Load models
    logger.info("Loading models...")
    ae_model = AutoencoderAnomalyDetector(
        sector='autoencoder',
        model_dir=model_dir
    )
    ae_model.load()
    
    if_model = IsolationForestAnomalyDetector(
        sector='isolation_forest',
        model_dir=model_dir
    )
    if_model.load()
    logger.info("✓ Models loaded")
    
    # Initialize feature extractor
    extractor = MarketFeatureExtractor(str(db_path))
    
    # Evaluate training set
    training_scores = []
    training_aggregate = {}
    
    if not args.skip_training:
        training_scores, training_aggregate = evaluate_training_set(
            extractor, db_path, ae_model, if_model
        )
    else:
        logger.info("Skipping training set evaluation (--skip-training)")
    
    # Evaluate portfolio
    portfolio_scores, portfolio_aggregate = evaluate_portfolio(
        portfolio_symbols,
        portfolio_shares if portfolio_shares else None,
        extractor,
        db_path,
        ae_model,
        if_model,
        training_scores=training_scores if not args.skip_training else None
    )
    
    if not portfolio_scores:
        logger.error("Failed to score any portfolio securities")
        sys.exit(1)
    
    # Print results
    if not args.skip_training:
        print_results(
            training_scores,
            training_aggregate,
            portfolio_scores,
            portfolio_aggregate,
            ae_model
        )
    else:
        # Simplified output when skipping training set
        print("\n" + "="*100)
        print("PORTFOLIO EVALUATION RESULTS")
        print("="*100)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("PORTFOLIO SUMMARY")
        print("-"*100)
        print(f"Number of securities: {portfolio_aggregate['n_securities']}")
        print(f"Autoencoder weighted score: {portfolio_aggregate['ae_weighted_score']:.6f}")
        if 'ae_weighted_z_score' in portfolio_aggregate and portfolio_aggregate['ae_weighted_z_score'] is not None:
            print(f"Autoencoder weighted z-score: {portfolio_aggregate['ae_weighted_z_score']:+.3f} (vs basket)")
        print(f"Autoencoder threshold: {ae_model.threshold:.6f}")
        print(f"Isolation Forest weighted score: {portfolio_aggregate['if_weighted_score']:.6f}")
        if 'if_weighted_z_score' in portfolio_aggregate and portfolio_aggregate['if_weighted_z_score'] is not None:
            print(f"Isolation Forest weighted z-score: {portfolio_aggregate['if_weighted_z_score']:+.3f} (vs basket)")
        print(f"Anomaly detection rate (AE): {portfolio_aggregate['anomaly_rate_ae']*100:.2f}%")
        print(f"Anomaly detection rate (IF): {portfolio_aggregate['anomaly_rate_if']*100:.2f}%")
        print()
        print("PORTFOLIO SECURITIES DETAIL")
        print("-"*100)
        has_shares = any('shares' in s for s in portfolio_scores)
        has_z_scores = any(s.get('if_score_z_score') is not None for s in portfolio_scores)
        weights = portfolio_aggregate.get('weights', {})
        for score in portfolio_scores:
            weight = weights.get(score['symbol'], 0) * 100
            if has_shares and 'shares' in score:
                shares = score.get('shares', 0)
                value = score.get('position_value', 0)
                z_info = ""
                if has_z_scores:
                    if_z = score.get('if_score_z_score', None)
                    ae_z = score.get('ae_score_z_score', None)
                    if_z_str = f", IF_z={if_z:+.2f}" if if_z is not None else ""
                    ae_z_str = f", AE_z={ae_z:+.2f}" if ae_z is not None else ""
                    z_info = f"{if_z_str}{ae_z_str}"
                print(f"{score['symbol']}: {shares:.2f} shares, {weight:.2f}% weight, ${value:.2f} value, "
                      f"AE={score['ae_score']:.6f} (anomaly: {score['ae_is_anomaly']}), "
                      f"IF={score['if_score']:.6f} (anomaly: {score['if_is_anomaly']}){z_info}")
            else:
                z_info = ""
                if has_z_scores:
                    if_z = score.get('if_score_z_score', None)
                    ae_z = score.get('ae_score_z_score', None)
                    if_z_str = f", IF_z={if_z:+.2f}" if if_z is not None else ""
                    ae_z_str = f", AE_z={ae_z:+.2f}" if ae_z is not None else ""
                    z_info = f"{if_z_str}{ae_z_str}"
                print(f"{score['symbol']}: {weight:.2f}% weight, "
                      f"AE={score['ae_score']:.6f} (anomaly: {score['ae_is_anomaly']}), "
                      f"IF={score['if_score']:.6f} (anomaly: {score['if_is_anomaly']}){z_info}")
        print("="*100)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_symbols': portfolio_symbols,
            'portfolio_shares': portfolio_shares if portfolio_shares else None,
            'portfolio_weights': portfolio_aggregate.get('weights', {}),
            'portfolio_scores': portfolio_scores,
            'portfolio_aggregate': portfolio_aggregate,
            'training_scores': training_scores if not args.skip_training else None,
            'training_aggregate': training_aggregate if not args.skip_training else None,
            'z_score_available': any(s.get('if_score_z_score') is not None for s in portfolio_scores) if portfolio_scores else False
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

