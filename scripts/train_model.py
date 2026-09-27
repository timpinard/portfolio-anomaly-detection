#!/usr/bin/env python3
"""
Train model of a specific type.

Usage:
    # Train with all symbols in database (default)
    python scripts/train_model.py individual
    python scripts/train_model.py cross_sectional
    
    # Train with custom universe (~60 symbols)
    python scripts/train_model.py individual --universe custom
    python scripts/train_model.py cross_sectional --universe custom
    
    # Train with S&P 500 symbols (if available in database)
    python scripts/train_model.py individual --universe sp500
    
    # Overwrite cached features and use custom universe
    python scripts/train_model.py cross_sectional --universe custom --overwrite-features
    
    # Train with date filtering (recommended for temporal validation)
    python scripts/train_model.py cross_sectional --train-start 2021-01-01 --train-end 2024-12-31 --universe custom
    
    # Use parallel extraction with custom worker count (cross_sectional only)
    python scripts/train_model.py cross_sectional --universe custom --workers 8
    
    # List available model types
    python scripts/train_model.py --list
    
Note:
    - cross_sectional models automatically use parallel feature extraction
    - individual models use sequential extraction
    - Parallel extraction significantly speeds up cross_sectional feature calculation
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import logging
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sqlite3
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import get_feature_extractor, FeatureExtractorFactory
from features.storage import FeatureStore
from models.autoencoder import AutoencoderAnomalyDetector
from data.universe import get_custom_universe
from utils.visualizations import (
    plot_data_split, plot_training_loss, plot_training_score_distribution,
    plot_feature_correlation
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_type_config(model_type: str) -> dict:
    """Load model type configuration from merged model_config.yaml."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'model_config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    model_types = full_config.get('model_types', {})
    
    if model_type not in model_types:
        available = list(model_types.keys())
        raise FileNotFoundError(
            f"Model type '{model_type}' not found in config.\n"
            f"Available model types: {available}"
        )
    
    return model_types[model_type]


def list_model_types() -> list:
    """List available model type configurations."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'model_config.yaml'

    if not config_path.exists():
        return []

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    
    model_types = full_config.get('model_types', {})
    return list(model_types.keys())


def prepare_feature_matrix(df, scaler=None, fit_scaler=True):
    """Prepare feature matrix with optional standardization."""
    exclude_cols = {'symbol', 'date', 'index'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X = scaler.fit_transform(X)
        logger.info(f"Fitted StandardScaler on {X.shape[0]} samples")
    else:
        X = scaler.transform(X)

    return X, feature_cols, scaler


def process_symbol_parallel(args):
    """Process a single symbol - runs in worker process for parallel extraction."""
    symbol, db_path, extractor_type, extractor_config = args

    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from features import get_feature_extractor

    try:
        extractor = get_feature_extractor(extractor_type, db_path, extractor_config)
        df = extractor.load_price_data(symbol)

        if df.empty:
            return symbol, None, "No data"

        if extractor_type == 'cross_sectional':
            market_returns = extractor._load_market_returns()
            universe_returns = None  # Skip universe for speed, can add later
            features = extractor.calculate_features(df, market_returns=market_returns, universe_returns=universe_returns)
        else:
            features = extractor.calculate_features(df)

        features = features.dropna()

        if features.empty:
            return symbol, None, "No features after dropna"

        return symbol, features, None

    except Exception as e:
        return symbol, None, str(e)


def extract_features_parallel(
    db_path: Path,
    extractor_type: str,
    extractor_config: dict,
    symbols: list[str] = None,
    num_workers: int = None,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Extract features in parallel for faster processing.
    
    Args:
        db_path: Path to database
        extractor_type: Type of feature extractor
        extractor_config: Configuration for extractor
        symbols: List of symbols to process (None = all)
        num_workers: Number of worker processes (None = CPU count)
        batch_size: Batch size for progress reporting
    
    Returns:
        DataFrame with all features
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    if symbols is None:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
    
    if extractor_type == 'cross_sectional':
        market_proxy = extractor_config.get('market_proxy', 'SPY')
        symbols = [s for s in symbols if s != market_proxy]
    
    if not symbols:
        logger.warning("No symbols to process")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(symbols)} symbols with {num_workers} workers")
    logger.info(f"Extractor: {extractor_type}")
    
    work_items = [(s, str(db_path), extractor_type, extractor_config) for s in symbols]
    
    start_time = time.time()
    completed = 0
    errors = 0
    all_features = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_symbol_parallel, item): item[0] for item in work_items}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, features, error = future.result()
                
                if error:
                    errors += 1
                    logger.warning(f"  {sym}: {error}")
                elif features is not None:
                    all_features.append(features)
                    completed += 1
                    
                    if completed % batch_size == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed * 60 if elapsed > 0 else 0
                        remaining = (len(symbols) - completed - errors) / rate if rate > 0 else 0
                        logger.info(f"Progress: {completed}/{len(symbols)} done, "
                                   f"{rate:.1f}/min, ~{remaining:.1f} min remaining")
                
            except Exception as e:
                errors += 1
                logger.error(f"  {symbol}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Parallel extraction complete: {completed} symbols in {elapsed/60:.1f} minutes")
    if errors > 0:
        logger.warning(f"Errors: {errors}")
    
    if not all_features:
        logger.error("No features extracted!")
        return pd.DataFrame()
    
    combined = pd.concat(all_features, axis=0)
    
    if 'date' in combined.columns:
        # If date is in index, move it to column
        if isinstance(combined.index, pd.DatetimeIndex):
            combined = combined.reset_index()
            if 'index' in combined.columns:
                combined.rename(columns={'index': 'date'}, inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(combined['date']):
            combined['date'] = pd.to_datetime(combined['date'])
        if combined['date'].dt.tz is not None:
            combined['date'] = combined['date'].dt.tz_localize(None)
    
    logger.info(f"Total feature rows: {len(combined):,}")
    
    return combined


def get_universe_symbols(universe: str, db_path: Path) -> list[str]:
    """
    Get list of symbols based on universe type.
    
    Args:
        universe: Universe type ('all', 'custom', or 'sp500')
        db_path: Path to database for querying available symbols
    
    Returns:
        List of symbol strings
    """
    if universe == 'custom':
        symbols = get_custom_universe()
        logger.info(f"Using custom universe: {len(symbols)} symbols")
        return symbols
    elif universe == 'sp500':
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM universe_sp500 ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        if not symbols:
            logger.warning("No S&P 500 symbols found in database. Falling back to all symbols.")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
        logger.info(f"Using S&P 500 universe: {len(symbols)} symbols")
        return symbols
    else:  # 'all'
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"Using all symbols in database: {len(symbols)} symbols")
        return symbols


def train_model(
    model_type: str,
    overwrite_features: bool = False,
    train_start: str = None,
    train_end: str = None,
    universe: str = 'all',
    num_workers: int = None
):
    """Train models for a specific model type.

    Args:
        model_type: Name of the model type config
        overwrite_features: Whether to recalculate features
        train_start: Start date for training data (YYYY-MM-DD), None for earliest
        train_end: End date for training data (YYYY-MM-DD), None for latest
        universe: Universe type ('all', 'custom', or 'sp500')
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING MODEL TYPE: {model_type}")
    logger.info("=" * 80)

    config = load_model_type_config(model_type)
    exp_config = config['model_type']
    feature_config = config['features']
    model_config = config['models']

    data_config = config.get('data', {})
    if train_start is None:
        train_start = data_config.get('train_start_date')
    if train_end is None:
        train_end = data_config.get('train_end_date')
    
    if train_end is None:
        logger.warning("=" * 80)
        logger.warning("WARNING: No train_end_date specified!")
        logger.warning("=" * 80)
        logger.warning("Training will use ALL available data up to the latest date.")
        logger.warning("This means there will be NO test data for temporal validation.")
        logger.warning("")
        logger.warning("Recommendation: Specify --train-end to leave data for testing.")
        logger.warning("Example: --train-end 2024-12-31")
        logger.warning("")
        logger.warning("Continuing with all available data...")
        logger.warning("=" * 80)
    
    logger.info(f"Date ranges: train_start={train_start}, train_end={train_end}")
    if train_start or train_end:
        logger.info(f"  (from config: train_start_date={data_config.get('train_start_date')}, train_end_date={data_config.get('train_end_date')})")

    logger.info(f"Description: {exp_config['description']}")

    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'model_types' / model_type

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run 'python scripts/fetch_data.py' first")
        sys.exit(1)

    model_dir.mkdir(parents=True, exist_ok=True)
    training_viz_dir = model_dir / 'training_visualizations'
    training_viz_dir.mkdir(parents=True, exist_ok=True)

    feature_store = FeatureStore(str(db_path))

    symbols = get_universe_symbols(universe, db_path) if universe != 'all' else None

    extractor_type = feature_config['extractor']
    extractor_config = feature_config.get('config', {})

    if feature_store.table_exists(model_type) and not overwrite_features:
        logger.info("\n" + "-" * 80)
        logger.info("LOADING CACHED FEATURES")
        logger.info("-" * 80)
        features_df = feature_store.load_features(model_type)
        logger.info(f"Loaded {len(features_df):,} cached feature rows")
        
        if 'date' in features_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                features_df['date'] = pd.to_datetime(features_df['date'])
            if features_df['date'].dt.tz is not None:
                features_df['date'] = features_df['date'].dt.tz_localize(None)
        
        if symbols is not None:
            initial_count = len(features_df)
            if 'symbol' in features_df.columns:
                features_df = features_df[features_df['symbol'].isin(symbols)]
                logger.info(f"Filtered to {len(features_df):,} rows for {len(symbols)} symbols")
                if len(features_df) == 0:
                    logger.warning("No features found for specified symbols. Consider using --overwrite-features")
            else:
                logger.warning("Cached features don't have 'symbol' column. Cannot filter by universe.")
    else:
        logger.info("\n" + "-" * 80)
        logger.info("EXTRACTING FEATURES")
        logger.info("-" * 80)

        if overwrite_features:
            logger.info("(--overwrite-features flag set)")

        logger.info(f"Using feature extractor: {extractor_type}")
        
        if extractor_type == 'cross_sectional':
            logger.info("Using parallel feature extraction for cross_sectional model")
            if num_workers is None:
                num_workers = mp.cpu_count()
            logger.info(f"Using {num_workers} worker processes")
            features_df = extract_features_parallel(
                db_path=db_path,
                extractor_type=extractor_type,
                extractor_config=extractor_config,
                symbols=symbols,
                num_workers=num_workers,
                batch_size=10
            )
        else:
            extractor = get_feature_extractor(extractor_type, str(db_path), extractor_config)
            features_df = extractor.prepare_training_data(symbols=symbols)

        if features_df.empty:
            logger.error("No features extracted!")
            sys.exit(1)

        feature_store.save_features(model_type, features_df, overwrite=True)
        logger.info(f"Cached {len(features_df):,} feature rows")

    logger.info("\n" + "-" * 80)
    logger.info("FILTERING TRAINING DATA")
    logger.info("-" * 80)

    date_in_column = 'date' in features_df.columns
    date_in_index = isinstance(features_df.index, pd.DatetimeIndex)
    
    if not date_in_column and date_in_index:
        features_df = features_df.reset_index()
        if 'index' in features_df.columns:
            features_df.rename(columns={'index': 'date'}, inplace=True)
        date_in_column = True
    
    if date_in_column:
        if not pd.api.types.is_datetime64_any_dtype(features_df['date']):
            features_df['date'] = pd.to_datetime(features_df['date'])
        
        if features_df['date'].dt.tz is not None:
            features_df['date'] = features_df['date'].dt.tz_localize(None)

        all_data_start = features_df['date'].min().strftime('%Y-%m-%d')
        all_data_end = features_df['date'].max().strftime('%Y-%m-%d')
        logger.info(f"Full data range: {all_data_start} to {all_data_end}")

        if train_start:
            train_start_dt = pd.to_datetime(train_start)
            if train_start_dt.tz is not None:
                train_start_dt = train_start_dt.tz_localize(None)
            features_df = features_df[features_df['date'] >= train_start_dt]
            logger.info(f"Train start filter: >= {train_start}")
        if train_end:
            train_end_dt = pd.to_datetime(train_end)
            if train_end_dt.tz is not None:
                train_end_dt = train_end_dt.tz_localize(None)
            features_df = features_df[features_df['date'] <= train_end_dt]
            logger.info(f"Train end filter: <= {train_end}")

        actual_train_start = features_df['date'].min().strftime('%Y-%m-%d')
        actual_train_end = features_df['date'].max().strftime('%Y-%m-%d')
        logger.info(f"Training data range: {actual_train_start} to {actual_train_end}")
        logger.info(f"Training samples: {len(features_df):,}")
    else:
        actual_train_start = None
        actual_train_end = None
        logger.warning("No 'date' column or DatetimeIndex found - using all data")

    X, feature_cols, scaler = prepare_feature_matrix(features_df)
    logger.info(f"\nFeature matrix shape: {X.shape}")
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")
    
    if 'date' in features_df.columns and actual_train_end:
        logger.info("\nGenerating training visualizations...")
        try:
            plot_data_split(features_df, actual_train_end, 
                          output_path=training_viz_dir / 'data_split_timeline.png')
            logger.info("  Data split timeline")
        except Exception as e:
            logger.warning(f"  Failed to create data split timeline: {e}")
        
        try:
            plot_feature_correlation(features_df, feature_cols,
                                    output_path=training_viz_dir / 'feature_correlation.png')
            logger.info("  Feature correlation matrix")
        except Exception as e:
            logger.warning(f"  Failed to create feature correlation: {e}")

    joblib.dump(scaler, model_dir / 'scaler.joblib')
    joblib.dump(feature_cols, model_dir / 'feature_columns.joblib')
    logger.info(f"Saved scaler and feature columns")

    with open(model_dir / 'model_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    training_metadata = {
        'model_type': model_type,
        'trained_at': datetime.now().isoformat(),
        'train_start_date': actual_train_start,
        'train_end_date': actual_train_end,
        'train_samples': len(features_df),
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
    }
    with open(model_dir / 'training_metadata.json', 'w') as f:
        json.dump(training_metadata, f, indent=2)
    logger.info(f"Saved training metadata")

    logger.info("\n" + "-" * 80)
    logger.info("TRAINING AUTOENCODER")
    logger.info("-" * 80)

    ae_config = model_config['autoencoder']

    threshold_percentile = ae_config.get('threshold_percentile', 95)

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
        learning_rate=ae_config.get('learning_rate', 0.001),
        threshold_percentile=threshold_percentile
    )

    ae_model.save()
    logger.info(f"Autoencoder trained and saved")
    logger.info(f"  Threshold percentile: {threshold_percentile}")
    logger.info(f"  Threshold value: {ae_model.threshold:.6f}")

    scores = ae_model.score(X)
    predictions = ae_model.predict(X)
    anomaly_count = np.sum(predictions == -1)
    anomaly_rate = anomaly_count / len(predictions) * 100

    logger.info(f"  Training anomaly rate: {anomaly_rate:.2f}%")
    logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.info(f"  Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")
    
    try:
        plot_training_loss(ae_model.history, 
                          output_path=training_viz_dir / 'training_loss_curve.png')
        logger.info("  Training loss curve")
    except Exception as e:
        logger.warning(f"  Failed to create training loss curve: {e}")
    
    try:
        plot_training_score_distribution(scores, ae_model.threshold, threshold_percentile,
                                        output_path=training_viz_dir / 'training_score_distribution.png')
        logger.info("  Training score distribution")
    except Exception as e:
        logger.warning(f"  Failed to create training score distribution: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {model_type}")
    logger.info(f"Models saved to: {model_dir}")
    logger.info(f"Training visualizations saved to: {training_viz_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  Validate: python scripts/validate_model.py {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description='Train models for a model type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with all symbols in database (default)
    python scripts/train_model.py individual
    python scripts/train_model.py cross_sectional
    
    # Train with custom universe (~60 symbols)
    python scripts/train_model.py individual --universe custom
    python scripts/train_model.py cross_sectional --universe custom
    
    # Train with S&P 500 symbols (if available in database)
    python scripts/train_model.py individual --universe sp500
    
    # Overwrite cached features and use custom universe
    python scripts/train_model.py cross_sectional --universe custom --overwrite-features
    
    # Train with date filtering and custom universe
    python scripts/train_model.py cross_sectional --train-end 2024-01-01 --universe custom
    
    # Use parallel extraction with custom worker count (cross_sectional only)
    python scripts/train_model.py cross_sectional --universe custom --workers 8
    
    # List available model types
    python scripts/train_model.py --list
        """
    )

    parser.add_argument(
        'model_type',
        nargs='?',
        help='Name of the model type to train'
    )

    parser.add_argument(
        '--overwrite-features', '-o',
        action='store_true',
        help='Recalculate and overwrite cached features'
    )

    parser.add_argument(
        '--train-start',
        type=str,
        default=None,
        help='Training data start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--train-end',
        type=str,
        default=None,
        help='Training data end date (YYYY-MM-DD). Data after this is held out for validation.'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available model types'
    )

    parser.add_argument(
        '--universe', '-u',
        type=str,
        choices=['all', 'custom', 'sp500'],
        default='all',
        help='Universe of symbols to use for training. Options: all (default), custom (~60 symbols), sp500 (S&P 500 if available)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help=f'Number of worker processes for parallel extraction (default: CPU count, only used for cross_sectional models)'
    )

    args = parser.parse_args()

    if args.list:
        model_types = list_model_types()
        print("\nAvailable model types:")
        for exp in model_types:
            try:
                config = load_model_type_config(exp)
                desc = config['model_type'].get('description', 'No description')
                print(f"  {exp}: {desc}")
            except:
                print(f"  {exp}: (config error)")

        print("\nAvailable feature extractors:")
        for name, desc in FeatureExtractorFactory.list_extractors().items():
            print(f"  {name}: {desc}")
        return

    if not args.model_type:
        parser.print_help()
        sys.exit(1)

    train_model(
        args.model_type,
        overwrite_features=args.overwrite_features,
        train_start=args.train_start,
        train_end=args.train_end,
        universe=args.universe,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
