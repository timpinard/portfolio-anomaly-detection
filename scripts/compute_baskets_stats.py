#!/usr/bin/env python3
"""
Compute and save basket statistics for z-score calculation.

Run this script after training models to pre-compute the basket statistics
that enable z-score comparison in the API.

Usage:
    python compute_basket_stats.py
    
This will:
1. Load the trained models
2. Score all securities in the training set
3. Compute mean/std for both AE and IF scores
4. Save to models/market_universe/basket_stats.json
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector
from data.feature_extractor import MarketFeatureExtractor
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_training_symbols(db_path: Path) -> list:
    """Get symbols from the training set."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_features ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols


def compute_basket_stats(
    model_dir: Path,
    db_path: Path,
    output_path: Path
) -> dict:
    """
    Compute basket statistics from training data.
    
    Args:
        model_dir: Directory containing trained models
        db_path: Path to SQLite database
        output_path: Where to save the statistics
    
    Returns:
        Dictionary with computed statistics
    """
    logger.info("Loading models...")
    
    # Load models
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
    
    # Get training symbols
    symbols = get_training_symbols(db_path)
    logger.info(f"Found {len(symbols)} symbols in training set")
    
    # Initialize feature extractor
    extractor = MarketFeatureExtractor(str(db_path))
    
    # Score each symbol
    ae_scores = []
    if_scores = []
    scored_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(symbols)} symbols scored...")
        
        try:
            df = extractor.load_price_data(symbol)
            if df.empty or len(df) < 60:
                continue
            
            features = extractor.calculate_all_features(df)
            if features.empty:
                continue
            
            # Get most recent features
            latest = features.iloc[-1]
            feature_cols = extractor.get_feature_columns(features)
            X = np.array([[latest[col] for col in feature_cols]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Score with both models
            ae_score = float(ae_model.score(X)[0])
            if_score = float(if_model.score(X)[0])
            
            ae_scores.append(ae_score)
            if_scores.append(if_score)
            scored_symbols.append(symbol)
            
        except Exception as e:
            logger.debug(f"Error scoring {symbol}: {e}")
            continue
    
    logger.info(f"Successfully scored {len(ae_scores)}/{len(symbols)} symbols")
    
    if len(ae_scores) < 5:
        raise ValueError(f"Insufficient data: only {len(ae_scores)} symbols scored")
    
    # Compute statistics
    stats = {
        "ae_mean": float(np.mean(ae_scores)),
        "ae_std": float(np.std(ae_scores)),
        "ae_median": float(np.median(ae_scores)),
        "ae_min": float(np.min(ae_scores)),
        "ae_max": float(np.max(ae_scores)),
        "if_mean": float(np.mean(if_scores)),
        "if_std": float(np.std(if_scores)),
        "if_median": float(np.median(if_scores)),
        "if_min": float(np.min(if_scores)),
        "if_max": float(np.max(if_scores)),
        "n_securities": len(ae_scores),
        "computed_at": datetime.utcnow().isoformat(),
        "ae_threshold": float(ae_model.threshold),
        "symbols_scored": scored_symbols
    }
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ Saved basket statistics to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BASKET STATISTICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Securities scored: {stats['n_securities']}")
    logger.info(f"AE score - mean: {stats['ae_mean']:.6f}, std: {stats['ae_std']:.6f}")
    logger.info(f"AE score - range: [{stats['ae_min']:.6f}, {stats['ae_max']:.6f}]")
    logger.info(f"IF score - mean: {stats['if_mean']:.6f}, std: {stats['if_std']:.6f}")
    logger.info(f"IF score - range: [{stats['if_min']:.6f}, {stats['if_max']:.6f}]")
    logger.info(f"AE threshold: {stats['ae_threshold']:.6f}")
    logger.info("=" * 60)
    
    return stats


def main():
    """Main entry point."""
    # Setup paths
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'market_universe'
    output_path = model_dir / 'basket_stats.json'
    
    # Validate paths
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run 'make fetch-data' first")
        sys.exit(1)
    
    ae_path = model_dir / 'autoencoder' / 'model.pth'
    if_path = model_dir / 'isolation_forest' / 'model.joblib'
    
    if not ae_path.exists() or not if_path.exists():
        logger.error(f"Models not found in {model_dir}")
        logger.error("Run 'make train' first")
        sys.exit(1)
    
    # Compute and save statistics
    try:
        stats = compute_basket_stats(model_dir, db_path, output_path)
        logger.info("\n✓ Basket statistics computed successfully!")
        logger.info(f"  File: {output_path}")
        logger.info(f"  Securities: {stats['n_securities']}")
    except Exception as e:
        logger.error(f"Failed to compute basket statistics: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()