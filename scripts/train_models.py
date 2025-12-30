#!/usr/bin/env python3
"""Train universal anomaly detection models on market data."""

import sys
from pathlib import Path
import yaml
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(db_path: Path):
    """Load feature data from database."""
    conn = sqlite3.connect(db_path)
    
    try:
        df = pd.read_sql_query("SELECT * FROM market_features", conn)
    except sqlite3.OperationalError as e:
        logger.error(f"Error loading features: {e}")
        logger.error("Run 'python scripts/derive_features.py' first")
        conn.close()
        sys.exit(1)
    
    conn.close()
    return df


def prepare_feature_matrix(df: pd.DataFrame):
    """Prepare feature matrix for training."""
    # Exclude metadata columns
    exclude_cols = {'symbol', 'date'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    
    # Replace inf and NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, feature_cols


def archive_existing_model(model_path: Path, archive_base_dir: Path):
    """Archive existing model directory if it exists."""
    if model_path.exists() and model_path.is_dir():
        # Create archive directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = archive_base_dir / f"{model_path.name}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the model directory to archive
        shutil.move(str(model_path), str(archive_dir / model_path.name))
        logger.info(f"✓ Archived existing {model_path.name} model to {archive_dir}")
        return True
    return False


def main():
    logger.info("="*80)
    logger.info("TRAINING UNIVERSAL MODELS")
    logger.info("="*80)
    
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'model_config.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Paths
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'market_universe'
    archive_dir = project_root / 'models' / 'archive'
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    
    # Archive existing models if they exist
    logger.info("\nChecking for existing models...")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    ae_model_path = model_dir / 'autoencoder'
    if_model_path = model_dir / 'isolation_forest'
    
    archived_ae = archive_existing_model(ae_model_path, archive_dir)
    archived_if = archive_existing_model(if_model_path, archive_dir)
    
    if not (archived_ae or archived_if):
        logger.info("No existing models found to archive")
    
    # Load training data
    logger.info("\nLoading training data...")
    df = load_training_data(db_path)
    logger.info(f"Loaded {len(df):,} feature rows from {df['symbol'].nunique()} symbols")
    logger.info(f"Using ALL feature records (no filters applied)")
    
    # Prepare features
    X, feature_cols = prepare_feature_matrix(df)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Train Autoencoder
    logger.info("\n" + "-"*80)
    logger.info("Training Autoencoder...")
    logger.info("-"*80)
    
    ae_config = config['models']['autoencoder']
    ae_model = AutoencoderAnomalyDetector(
        sector='autoencoder',
        model_dir=model_dir,
        input_dim=X.shape[1],
        encoding_dim=ae_config.get('encoding_dim', 15),
        hidden_dims=ae_config.get('hidden_dims', [128, 64, 32]),
        device=ae_config.get('device', 'cuda')
    )
    
    ae_model.fit(
        X,
        epochs=ae_config.get('epochs', 100),
        batch_size=ae_config.get('batch_size', 64),
        learning_rate=ae_config.get('learning_rate', 0.001)
    )
    
    ae_model.save()
    logger.info(f"✓ Autoencoder trained and saved")
    logger.info(f"  Threshold: {ae_model.threshold:.6f}")
    
    # Train Isolation Forest
    logger.info("\n" + "-"*80)
    logger.info("Training Isolation Forest...")
    logger.info("-"*80)
    
    if_config = config['models']['isolation_forest']
    # Handle max_samples: None/null means use all samples, 'auto' means min(256, n_samples)
    max_samples = if_config.get('max_samples', 'auto')
    if max_samples is None or (isinstance(max_samples, str) and max_samples.lower() == 'null'):
        max_samples = None  # Use all samples in each tree
    
    if_model = IsolationForestAnomalyDetector(
        sector='isolation_forest',
        model_dir=model_dir,
        contamination=if_config.get('contamination', 0.05),
        n_estimators=if_config.get('n_estimators', 100),
        max_samples=max_samples,
        random_state=if_config.get('random_state', 42)
    )
    
    if_model.fit(X)
    if_model.save()
    
    # Check anomaly rate
    predictions = if_model.predict(X)
    anomaly_count = np.sum(predictions == -1)
    anomaly_rate = anomaly_count / len(predictions) * 100
    
    logger.info(f"✓ Isolation Forest trained and saved")
    logger.info(f"  Detected {anomaly_count} anomalies ({anomaly_rate:.2f}%)")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("✓ MODEL TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Models saved to: {model_dir}")
    logger.info(f"  - Autoencoder: {model_dir / 'autoencoder'}")
    logger.info(f"  - Isolation Forest: {model_dir / 'isolation_forest'}")
    logger.info("\nNext steps:")
    logger.info("  1. Start API: python api/service.py")
    logger.info("  2. Test: python api/test_client.py")


if __name__ == '__main__':
    main()
