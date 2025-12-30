#!/usr/bin/env python3
"""Derive features from market price data."""

import sys
from pathlib import Path
import sqlite3
import logging
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.feature_extractor import MarketFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("DERIVING MARKET FEATURES")
    logger.info("="*80)
    
    # Paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run 'python scripts/fetch_data.py' first")
        sys.exit(1)
    
    # Initialize extractor
    extractor = MarketFeatureExtractor(str(db_path))
    
    # Calculate features for all symbols
    features = extractor.prepare_training_data()
    
    if features.empty:
        logger.error("No features calculated!")
        sys.exit(1)
    
    # Save to database
    logger.info("\nSaving features to database...")
    conn = sqlite3.connect(db_path)
    
    # Drop existing table if it exists
    conn.execute("DROP TABLE IF EXISTS market_features")
    
    # Save features
    features.reset_index(inplace=True)  # Make date a column
    
    # Convert ALL datetime/timestamp columns to strings
    # SQLite doesn't support pandas Timestamp objects, so convert to strings
    for col in features.columns:
        # Check if column contains datetime-like objects
        if pd.api.types.is_datetime64_any_dtype(features[col]):
            # Convert datetime64 to string
            features[col] = features[col].astype(str)
        # Also check for object dtype columns that might contain Timestamps
        elif features[col].dtype == 'object':
            # Check if first non-null value is a Timestamp
            sample = features[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], pd.Timestamp):
                # Convert Timestamp objects to strings
                features[col] = features[col].apply(
                    lambda x: str(x) if isinstance(x, pd.Timestamp) else x
                )
    
    features.to_sql('market_features', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()
    
    logger.info("\n" + "="*80)
    logger.info("✓ FEATURE DERIVATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Table: market_features")
    logger.info(f"Total rows: {len(features)}")
    logger.info(f"Features per symbol: ~{len(features) // features['symbol'].nunique()}")
    logger.info(f"Feature columns: {len(extractor.get_feature_columns(features))}")


if __name__ == '__main__':
    main()