#!/usr/bin/env python3
"""
Unified validation script for anomaly detection models.

This script supports two validation modes:

1. TEMPORAL BACKTEST (default):
   - Uses data AFTER the training period for testing
   - Tests if high scores predict unusual forward returns/volatility
   - Most realistic validation approach
   - Works for both individual and cross_sectional model types

2. SYNTHETIC ANOMALY VALIDATION (--synthetic flag):
   - Creates synthetic anomalies to test detection capabilities
   - Uses 80/20 train/test split
   - Useful for validating model sensitivity to known patterns
   - Works for both individual and cross_sectional model types

Usage:
    # Temporal backtest (default)
    python scripts/validate_model.py individual
    python scripts/validate_model.py cross_sectional
    python scripts/validate_model.py cross_sectional --test-start 2024-01-01
    python scripts/validate_model.py individual --top-k 20

    # Synthetic anomaly validation
    python scripts/validate_model.py individual --synthetic
    python scripts/validate_model.py cross_sectional --synthetic
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import sqlite3
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import get_feature_extractor
from features.storage import FeatureStore
from models.autoencoder import AutoencoderAnomalyDetector
from utils.visualizations import (
    plot_test_score_distribution, plot_score_time_series, plot_anomaly_rate_by_period,
    plot_persistent_anomalies_heatmap, plot_forward_return_comparison,
    plot_score_vs_forward_metrics, plot_reconstruction_examples,
    plot_anomalous_reconstructions, create_validation_dashboard
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_metadata(model_dir: Path) -> dict:
    """Load training metadata to determine holdout period."""
    metadata_path = model_dir / 'training_metadata.json'
    if not metadata_path.exists():
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

    fwd_vol = daily_returns.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    fwd_vol_stacked = fwd_vol.stack().reset_index()
    fwd_vol_stacked.columns = ['date', 'symbol', f'fwd_volatility_{horizon}d']

    return fwd_vol_stacked


def create_divergence_anomalies(X_normal, n_anomalies=100, feature_names=None):
    """
    Create synthetic anomalies specific to cross_sectional/divergence detection.

    These simulate stocks that are moving opposite to the market:
    1. Negative relative returns (stock down, market up)
    2. Low/negative correlation
    3. Negative beta
    4. Extreme z-scores vs universe
    """
    n_features = X_normal.shape[1]
    mean = np.mean(X_normal, axis=0)
    std = np.std(X_normal, axis=0)
    std = np.where(std < 1e-8, 1e-8, std)

    anomalies = []

    feat_idx = {}
    if feature_names:
        for i, name in enumerate(feature_names):
            feat_idx[name] = i

    n_per_type = n_anomalies // 4

    # Type 1: Strong negative divergence (stock down while market up)
    for _ in range(n_per_type):
        anomaly = mean.copy()
        if feat_idx:
            # Negative relative returns
            for col in ['relative_return_1d', 'relative_return_5d', 'relative_return_20d']:
                if col in feat_idx:
                    anomaly[feat_idx[col]] = mean[feat_idx[col]] - np.random.uniform(3, 5) * std[feat_idx[col]]

            # Low direction agreement
            if 'direction_agreement_20d' in feat_idx:
                anomaly[feat_idx['direction_agreement_20d']] = np.random.uniform(0.1, 0.3)

            # Negative z-score vs universe
            if 'zscore_vs_universe' in feat_idx:
                anomaly[feat_idx['zscore_vs_universe']] = np.random.uniform(-4, -2)
        else:
            # Fallback: perturb first few features
            for i in range(min(5, n_features)):
                anomaly[i] = mean[i] - np.random.uniform(3, 5) * std[i]

        anomalies.append(anomaly)

    # Type 2: Correlation breakdown (sudden decorrelation)
    for _ in range(n_per_type):
        anomaly = mean.copy()
        if feat_idx:
            # Low correlation
            for col in ['correlation_20d', 'correlation_60d']:
                if col in feat_idx:
                    anomaly[feat_idx[col]] = np.random.uniform(-0.3, 0.2)

            # Large correlation change
            for col in ['correlation_change_20d', 'correlation_change_60d']:
                if col in feat_idx:
                    anomaly[feat_idx[col]] = mean[feat_idx[col]] - np.random.uniform(2, 4) * std[feat_idx[col]]
        else:
            for i in range(n_features // 3, 2 * n_features // 3):
                anomaly[i] = mean[i] - np.random.uniform(2, 4) * std[i]

        anomalies.append(anomaly)

    # Type 3: Beta shift (becoming defensive/negative beta)
    for _ in range(n_per_type):
        anomaly = mean.copy()
        if feat_idx:
            # Low or negative beta
            if 'beta_60d' in feat_idx:
                anomaly[feat_idx['beta_60d']] = np.random.uniform(-0.5, 0.3)

            # High beta deviation
            if 'beta_deviation_60d' in feat_idx:
                anomaly[feat_idx['beta_deviation_60d']] = np.random.uniform(1.0, 2.0)

            # Beta change
            if 'beta_change_60d' in feat_idx:
                anomaly[feat_idx['beta_change_60d']] = mean[feat_idx['beta_change_60d']] - np.random.uniform(2, 4) * std[feat_idx['beta_change_60d']]
        else:
            for i in range(2 * n_features // 3, n_features):
                anomaly[i] = mean[i] - np.random.uniform(2, 4) * std[i]

        anomalies.append(anomaly)

    # Type 4: Extreme outperformance (stock way up, different from universe)
    remaining = n_anomalies - len(anomalies)
    for _ in range(remaining):
        anomaly = mean.copy()
        if feat_idx:
            # Positive relative returns (outperforming)
            for col in ['relative_return_1d', 'relative_return_5d', 'relative_return_20d']:
                if col in feat_idx:
                    anomaly[feat_idx[col]] = mean[feat_idx[col]] + np.random.uniform(3, 5) * std[feat_idx[col]]

            # High rank
            if 'return_rank_daily' in feat_idx:
                anomaly[feat_idx['return_rank_daily']] = np.random.uniform(0.95, 1.0)

            # Positive z-score vs universe
            if 'zscore_vs_universe' in feat_idx:
                anomaly[feat_idx['zscore_vs_universe']] = np.random.uniform(2, 4)
        else:
            for i in range(min(5, n_features)):
                anomaly[i] = mean[i] + np.random.uniform(3, 5) * std[i]

        anomalies.append(anomaly)

    return np.array(anomalies), np.ones(len(anomalies))


def create_generic_anomalies(X_normal, n_anomalies=100, feature_names=None):
    """Create generic synthetic anomalies (for individual model types)."""
    n_features = X_normal.shape[1]
    mean = np.mean(X_normal, axis=0)
    std = np.std(X_normal, axis=0)
    std = np.where(std < 1e-8, 1e-8, std)

    anomalies = []

    for _ in range(n_anomalies):
        anomaly = mean.copy()
        n_perturb = np.random.randint(3, 6)
        features_to_perturb = np.random.choice(n_features, n_perturb, replace=False)
        for feat_idx in features_to_perturb:
            direction = np.random.choice([-1, 1])
            magnitude = np.random.uniform(3, 5)
            anomaly[feat_idx] = mean[feat_idx] + direction * magnitude * std[feat_idx]
        anomalies.append(anomaly)

    return np.array(anomalies), np.ones(len(anomalies))


def evaluate_model(model, X_test, y_test, model_name, threshold=None):
    """Evaluate model performance."""
    scores = model.score(X_test)
    if threshold is None:
        threshold = model.threshold
    y_pred = (scores > threshold).astype(int)

    metrics = {
        'model_name': model_name,
        'n_samples': len(X_test),
        'n_anomalies_detected': int(np.sum(y_pred)),
        'anomaly_rate': float(np.mean(y_pred)),
    }

    if len(np.unique(y_test)) > 1:
        metrics['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_test, y_pred, zero_division=0))

        try:
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            metrics['roc_auc'] = float(roc_auc_score(y_test, scores_norm))
        except:
            metrics['roc_auc'] = None

        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
        }

    return metrics


def analyze_score_by_period(scores_df: pd.DataFrame, period_col: str = 'year_month') -> pd.DataFrame:
    """Analyze score distribution by time period."""
    scores_df[period_col] = scores_df['date'].dt.to_period('M')

    period_stats = scores_df.groupby(period_col).agg({
        'score': ['mean', 'std', 'median', lambda x: np.percentile(x, 95)],
        'symbol': 'count'
    }).round(4)

    period_stats.columns = ['mean_score', 'std_score', 'median_score', 'p95_score', 'n_samples']
    return period_stats.reset_index()


def get_top_anomalies_by_date(scores_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Get top K most anomalous stocks for each date."""
    top_anomalies = (
        scores_df
        .sort_values(['date', 'score'], ascending=[True, False])
        .groupby('date')
        .head(top_k)
    )
    return top_anomalies


def validate_synthetic(
    model_type: str,
    n_anomalies: int = 100
):
    """Run synthetic anomaly validation."""
    logger.info("=" * 80)
    logger.info(f"SYNTHETIC ANOMALY VALIDATION: {model_type}")
    logger.info("=" * 80)

    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'model_types' / model_type
    results_dir = project_root / 'results' / 'model_types' / model_type

    if not model_dir.exists():
        logger.error(f"Models not found: {model_dir}")
        logger.error(f"Run 'python scripts/train_model.py {model_type}' first")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_root / 'config' / 'model_config.yaml'
    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    
    model_config = full_config.get('model_types', {}).get(model_type, {})
    feature_config = model_config.get('features', {})
    extractor_type = feature_config.get('extractor', 'individual')

    logger.info("\nLoading model...")
    ae_model = AutoencoderAnomalyDetector.load(model_dir / 'autoencoder')
    scaler = joblib.load(model_dir / 'scaler.joblib')
    feature_cols = joblib.load(model_dir / 'feature_columns.joblib')

    feature_store = FeatureStore(str(db_path))
    features_df = feature_store.load_features(model_type)
    
    exclude_cols = {'symbol', 'date', 'index'}
    actual_feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X = features_df[actual_feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = scaler.transform(X)

    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]

    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Create appropriate synthetic anomalies based on model type
    logger.info("\nGenerating synthetic anomalies...")
    if extractor_type == 'cross_sectional':
        X_synthetic, y_synthetic = create_divergence_anomalies(
            X_train, n_anomalies=n_anomalies, feature_names=actual_feature_cols
        )
        anomaly_type = "divergence-specific"
    else:
        X_synthetic, y_synthetic = create_generic_anomalies(
            X_train, n_anomalies=n_anomalies, feature_names=actual_feature_cols
        )
        anomaly_type = "generic"

    logger.info(f"Created {len(X_synthetic)} {anomaly_type} anomalies")

    X_test_with_anomalies = np.vstack([X_test, X_synthetic])
    y_test_with_anomalies = np.hstack([np.zeros(len(X_test)), y_synthetic])

    logger.info("\nEvaluating model...")
    ae_metrics = evaluate_model(ae_model, X_test_with_anomalies, y_test_with_anomalies, "Autoencoder", ae_model.threshold)

    ae_synthetic_scores = ae_model.score(X_synthetic)
    ae_synthetic_pred = (ae_synthetic_scores > ae_model.threshold).astype(int)
    ae_synthetic_rate = float(np.mean(ae_synthetic_pred))

    all_scores = ae_model.score(X_test_with_anomalies)
    score_stats = {
        'mean': float(np.mean(all_scores)),
        'std': float(np.std(all_scores)),
        'min': float(np.min(all_scores)),
        'max': float(np.max(all_scores)),
        'threshold': float(ae_model.threshold)
    }

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Anomaly Type: {anomaly_type}")
    logger.info(f"Train Size: {len(X_train):,}")
    logger.info(f"Test Size: {len(X_test):,}")
    logger.info(f"Synthetic Anomalies: {len(X_synthetic):,}")
    logger.info(f"\nModel Metrics:")
    logger.info(f"  Anomaly Detection Rate: {ae_metrics['anomaly_rate']*100:.2f}%")
    if 'precision' in ae_metrics and ae_metrics['precision'] is not None:
        logger.info(f"  Precision: {ae_metrics['precision']:.4f}")
        logger.info(f"  Recall: {ae_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {ae_metrics['f1_score']:.4f}")
        if ae_metrics.get('roc_auc'):
            logger.info(f"  ROC-AUC: {ae_metrics['roc_auc']:.4f}")
        if ae_metrics.get('confusion_matrix'):
            cm = ae_metrics['confusion_matrix']
            logger.info(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")
    logger.info(f"\nSynthetic Anomaly Detection Rate: {ae_synthetic_rate*100:.1f}%")
    logger.info(f"\nScore Statistics:")
    logger.info(f"  Mean: {score_stats['mean']:.4f}")
    logger.info(f"  Std: {score_stats['std']:.4f}")
    logger.info(f"  Min: {score_stats['min']:.4f}")
    logger.info(f"  Max: {score_stats['max']:.4f}")
    logger.info(f"  Threshold: {score_stats['threshold']:.4f}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'model_type': model_type,
        'validation_mode': 'synthetic',
        'generated_at': timestamp,
        'anomaly_type': anomaly_type,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_synthetic_anomalies': len(X_synthetic),
        'model_metrics': ae_metrics,
        'synthetic_detection_rate': ae_synthetic_rate,
        'score_stats': score_stats
    }

    report_file = results_dir / f'synthetic_validation_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nSaved report to: {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Anomaly type: {anomaly_type}")
    logger.info(f"Synthetic detection rate: {ae_synthetic_rate*100:.1f}%")
    if 'precision' in ae_metrics and ae_metrics['precision'] is not None:
        logger.info(f"F1 Score: {ae_metrics['f1_score']:.4f}")
    logger.info("\nKey insight: High synthetic detection rate indicates the model can detect")
    logger.info("the specific anomaly patterns it was designed to find.")


def validate_model(
    model_type: str,
    test_start: str = None,
    test_end: str = None,
    top_k: int = 10,
    forward_horizons: list = [1, 5, 20]
):
    """Run temporal backtest validation."""
    logger.info("=" * 80)
    logger.info(f"TEMPORAL BACKTEST VALIDATION: {model_type}")
    logger.info("=" * 80)

    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'processed' / 'market_data.sqlite'
    model_dir = project_root / 'models' / 'model_types' / model_type
    results_dir = project_root / 'results' / 'model_types' / model_type

    if not model_dir.exists():
        logger.error(f"Models not found: {model_dir}")
        logger.error(f"Run 'python scripts/train_model.py {model_type}' first")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)
    validation_viz_dir = results_dir / 'validation_visualizations'
    validation_viz_dir.mkdir(parents=True, exist_ok=True)

    training_meta = load_training_metadata(model_dir)
    train_end_date = training_meta.get('train_end_date')

    logger.info(f"\nTraining period: {training_meta.get('train_start_date')} to {train_end_date}")
    logger.info(f"Training samples: {training_meta.get('train_samples', 'unknown'):,}")

    if test_start is None:
        if train_end_date:
            test_start = (pd.to_datetime(train_end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            logger.error("No train_end_date in metadata and no --test-start provided")
            sys.exit(1)

    logger.info(f"Test period: {test_start} to {test_end or 'latest'}")

    logger.info("\nLoading model...")
    ae_model = AutoencoderAnomalyDetector.load(model_dir / 'autoencoder')
    scaler = joblib.load(model_dir / 'scaler.joblib')
    feature_cols = joblib.load(model_dir / 'feature_columns.joblib')

    feature_store = FeatureStore(str(db_path))
    features_df = feature_store.load_features(model_type)
    features_df['date'] = pd.to_datetime(features_df['date'])
    if features_df['date'].dt.tz is not None:
        features_df['date'] = features_df['date'].dt.tz_localize(None)

    test_df = features_df[features_df['date'] >= test_start].copy()
    if test_end:
        test_df = test_df[test_df['date'] <= test_end]

    if test_df.empty:
        logger.error(f"No test data found for period {test_start} to {test_end}")
        sys.exit(1)

    logger.info(f"Test samples: {len(test_df):,}")
    logger.info(f"Test date range: {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}")
    logger.info(f"Unique symbols: {test_df['symbol'].nunique()}")
    logger.info(f"Unique dates: {test_df['date'].nunique()}")

    logger.info("\nScoring test data...")
    exclude_cols = {'symbol', 'date', 'index'}
    actual_feature_cols = [col for col in test_df.columns if col not in exclude_cols]

    X_test = test_df[actual_feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = scaler.transform(X_test)

    scores = ae_model.score(X_test)
    test_df['score'] = scores
    test_df['is_anomaly'] = (scores > ae_model.threshold).astype(int)
    
    logger.info("\nGenerating validation visualizations...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        train_mean = training_meta.get('mean_score') if 'mean_score' in training_meta else None
        plot_test_score_distribution(scores, ae_model.threshold, train_mean=train_mean,
                                    output_path=validation_viz_dir / f'test_score_distribution_{timestamp}.png')
        logger.info("  ✓ Test score distribution")
    except Exception as e:
        logger.warning(f"  ✗ Failed to create test score distribution: {e}")
    
    try:
        plot_score_time_series(test_df, ae_model.threshold,
                               output_path=validation_viz_dir / f'score_time_series_{timestamp}.png')
        logger.info("  ✓ Score time series")
    except Exception as e:
        logger.warning(f"  ✗ Failed to create score time series: {e}")
    
    try:
        plot_anomaly_rate_by_period(test_df,
                                    output_path=validation_viz_dir / f'anomaly_rate_by_month_{timestamp}.png')
        logger.info("  ✓ Anomaly rate by month")
    except Exception as e:
        logger.warning(f"  ✗ Failed to create anomaly rate by month: {e}")
    
    try:
        plot_persistent_anomalies_heatmap(test_df, top_n=20,
                                         output_path=validation_viz_dir / f'persistent_anomalies_heatmap_{timestamp}.png')
        logger.info("  ✓ Persistent anomalies heatmap")
    except Exception as e:
        logger.warning(f"  ✗ Failed to create persistent anomalies heatmap: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("SCORE DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)

    score_stats = {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'median': float(np.median(scores)),
        'p90': float(np.percentile(scores, 90)),
        'p95': float(np.percentile(scores, 95)),
        'p99': float(np.percentile(scores, 99)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'threshold': float(ae_model.threshold),
        'anomaly_rate': float(np.mean(test_df['is_anomaly']))
    }

    logger.info(f"Mean: {score_stats['mean']:.4f}")
    logger.info(f"Std: {score_stats['std']:.4f}")
    logger.info(f"Median: {score_stats['median']:.4f}")
    logger.info(f"90th percentile: {score_stats['p90']:.4f}")
    logger.info(f"95th percentile: {score_stats['p95']:.4f}")
    logger.info(f"99th percentile: {score_stats['p99']:.4f}")
    logger.info(f"Threshold: {score_stats['threshold']:.4f}")
    logger.info(f"Anomaly rate: {score_stats['anomaly_rate']*100:.2f}%")

    period_stats = analyze_score_by_period(test_df.copy())
    logger.info("\nScore by Month:")
    for _, row in period_stats.iterrows():
        logger.info(f"  {row['year_month']}: mean={row['mean_score']:.4f}, p95={row['p95_score']:.4f}, n={row['n_samples']:,}")

    logger.info("\n" + "=" * 80)
    logger.info(f"TOP {top_k} ANOMALIES (Most Recent Dates)")
    logger.info("=" * 80)

    recent_dates = sorted(test_df['date'].unique())[-5:]  # Last 5 dates

    for date in recent_dates:
        date_df = test_df[test_df['date'] == date].nlargest(top_k, 'score')
        logger.info(f"\n{date.strftime('%Y-%m-%d')}:")
        for _, row in date_df.iterrows():
            flag = "***" if row['is_anomaly'] else "   "
            logger.info(f"  {flag} {row['symbol']:6s}  score={row['score']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("FORWARD RETURN ANALYSIS")
    logger.info("=" * 80)
    logger.info("Do high anomaly scores predict unusual forward behavior?")

    symbols = test_df['symbol'].unique().tolist()
    prices_df = load_price_data(db_path, symbols, test_start)

    if not prices_df.empty:
        fwd_returns = calculate_forward_returns(prices_df, forward_horizons)
        fwd_vol = calculate_forward_volatility(prices_df)
        
        if fwd_returns['date'].dt.tz is not None:
            fwd_returns['date'] = fwd_returns['date'].dt.tz_localize(None)
        if fwd_vol['date'].dt.tz is not None:
            fwd_vol['date'] = fwd_vol['date'].dt.tz_localize(None)
        if test_df['date'].dt.tz is not None:
            test_df['date'] = test_df['date'].dt.tz_localize(None)

        analysis_df = test_df[['date', 'symbol', 'score', 'is_anomaly']].merge(
            fwd_returns, on=['date', 'symbol'], how='left'
        ).merge(
            fwd_vol, on=['date', 'symbol'], how='left'
        )
        
        try:
            plot_forward_return_comparison(analysis_df, forward_horizons,
                                          output_path=validation_viz_dir / f'forward_return_comparison_{timestamp}.png')
            logger.info("  ✓ Forward return comparison")
        except Exception as e:
            logger.warning(f"  ✗ Failed to create forward return comparison: {e}")
        
        try:
            plot_score_vs_forward_metrics(analysis_df, horizon=5,
                                          output_path=validation_viz_dir / f'score_vs_forward_return_{timestamp}.png')
            logger.info("  ✓ Score vs forward return")
        except Exception as e:
            logger.warning(f"  ✗ Failed to create score vs forward return: {e}")
        
        try:
            sample_indices = np.random.choice(len(X_test), min(100, len(X_test)), replace=False)
            sample_X = X_test[sample_indices]
            plot_reconstruction_examples(ae_model, sample_X, actual_feature_cols,
                                        output_path=validation_viz_dir / f'reconstruction_examples_{timestamp}.png')
            logger.info("  ✓ Reconstruction examples (bar chart)")
        except Exception as e:
            logger.warning(f"  ✗ Failed to create reconstruction examples: {e}")
        
        try:
            plot_anomalous_reconstructions(ae_model, X_test, actual_feature_cols,
                                         scores, ae_model.threshold, n_examples=3,
                                         output_path=validation_viz_dir / f'anomalous_reconstructions_{timestamp}.png')
            logger.info("  ✓ Anomalous reconstructions (line plot)")
        except Exception as e:
            logger.warning(f"  ✗ Failed to create anomalous reconstructions: {e}")

        for horizon in forward_horizons:
            col = f'fwd_return_{horizon}d'
            if col in analysis_df.columns:
                anomaly_returns = analysis_df[analysis_df['is_anomaly'] == 1][col].dropna()
                normal_returns = analysis_df[analysis_df['is_anomaly'] == 0][col].dropna()

                if len(anomaly_returns) > 0 and len(normal_returns) > 0:
                    logger.info(f"\n{horizon}-Day Forward Returns:")
                    logger.info(f"  Anomaly group (n={len(anomaly_returns):,}):")
                    logger.info(f"    Mean: {anomaly_returns.mean()*100:+.2f}%")
                    logger.info(f"    Std:  {anomaly_returns.std()*100:.2f}%")
                    logger.info(f"    Abs Mean: {anomaly_returns.abs().mean()*100:.2f}%")
                    logger.info(f"  Normal group (n={len(normal_returns):,}):")
                    logger.info(f"    Mean: {normal_returns.mean()*100:+.2f}%")
                    logger.info(f"    Std:  {normal_returns.std()*100:.2f}%")
                    logger.info(f"    Abs Mean: {normal_returns.abs().mean()*100:.2f}%")

        vol_col = 'fwd_volatility_20d'
        if vol_col in analysis_df.columns:
            anomaly_vol = analysis_df[analysis_df['is_anomaly'] == 1][vol_col].dropna()
            normal_vol = analysis_df[analysis_df['is_anomaly'] == 0][vol_col].dropna()

            if len(anomaly_vol) > 0 and len(normal_vol) > 0:
                logger.info(f"\n20-Day Forward Volatility:")
                logger.info(f"  Anomaly group: {anomaly_vol.mean()*100:.1f}% (n={len(anomaly_vol):,})")
                logger.info(f"  Normal group:  {normal_vol.mean()*100:.1f}% (n={len(normal_vol):,})")

        logger.info("\nCorrelation of Score with Forward Metrics:")
        for col in [f'fwd_return_{h}d' for h in forward_horizons] + ['fwd_volatility_20d']:
            if col in analysis_df.columns:
                valid = analysis_df[['score', col]].dropna()
                if len(valid) > 100:
                    corr = valid['score'].corr(valid[col].abs())
                    logger.info(f"  Score vs |{col}|: {corr:.4f}")
    else:
        analysis_df = None

    logger.info("\n" + "=" * 80)
    logger.info("PERSISTENT HIGH SCORERS")
    logger.info("=" * 80)
    logger.info("Stocks that consistently score high (structural anomalies)")

    top_anomalies = get_top_anomalies_by_date(test_df, top_k=top_k)
    symbol_freq = top_anomalies.groupby('symbol').size().sort_values(ascending=False)
    n_dates = test_df['date'].nunique()

    logger.info(f"\nSymbols appearing in top-{top_k} most frequently:")
    for symbol, count in symbol_freq.head(20).items():
        pct = count / n_dates * 100
        mean_score = test_df[test_df['symbol'] == symbol]['score'].mean()
        logger.info(f"  {symbol:6s}: {count:4d} times ({pct:5.1f}%), avg score={mean_score:.4f}")

    try:
        create_validation_dashboard(
            test_df, analysis_df, ae_model, training_meta, forward_horizons,
            output_path=validation_viz_dir / f'validation_dashboard_{timestamp}.png'
        )
        logger.info("  ✓ Validation dashboard")
    except Exception as e:
        logger.warning(f"  ✗ Failed to create validation dashboard: {e}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    scores_file = results_dir / f'backtest_scores_{timestamp}.csv'
    test_df[['date', 'symbol', 'score', 'is_anomaly']].to_csv(scores_file, index=False)
    logger.info(f"\nSaved scores to: {scores_file}")

    top_anomalies_file = results_dir / f'top_anomalies_{timestamp}.csv'
    top_anomalies.to_csv(top_anomalies_file, index=False)
    logger.info(f"Saved top anomalies to: {top_anomalies_file}")

    report = {
        'model_type': model_type,
        'generated_at': timestamp,
        'test_period': {
            'start': test_start,
            'end': test_end or test_df['date'].max().strftime('%Y-%m-%d'),
            'n_samples': len(test_df),
            'n_symbols': int(test_df['symbol'].nunique()),
            'n_dates': int(test_df['date'].nunique())
        },
        'training_period': {
            'start': training_meta.get('train_start_date'),
            'end': training_meta.get('train_end_date'),
            'n_samples': training_meta.get('train_samples')
        },
        'score_distribution': score_stats,
        'top_persistent_anomalies': symbol_freq.head(10).to_dict()
    }

    report_file = results_dir / f'backtest_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved report to: {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test period: {test_start} to {test_end or 'latest'}")
    logger.info(f"Anomaly rate: {score_stats['anomaly_rate']*100:.2f}%")
    logger.info(f"Most persistent anomaly: {symbol_freq.index[0]} ({symbol_freq.iloc[0]} times)")
    logger.info("\nKey insight: Compare forward return/volatility between anomaly and normal groups.")
    logger.info("If anomaly group shows higher absolute returns or volatility, the model is useful.")


def main():
    parser = argparse.ArgumentParser(
        description='Unified validation for anomaly detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script supports two validation modes:

1. TEMPORAL BACKTEST (default):
   - Tests on data AFTER training period (no data leakage)
   - Uses real market behavior
   - Measures if high scores predict unusual forward returns/volatility
   - Works for both individual and cross_sectional model types

2. SYNTHETIC ANOMALY VALIDATION (--synthetic flag):
   - Creates synthetic anomalies to test detection capabilities
   - Uses 80/20 train/test split
   - Useful for validating model sensitivity to known patterns
   - Works for both individual and cross_sectional model types

Examples:
    # Temporal backtest (default)
    python scripts/validate_model.py individual
    python scripts/validate_model.py cross_sectional
    python scripts/validate_model.py cross_sectional --test-start 2024-01-01
    python scripts/validate_model.py individual --top-k 20

    # Synthetic anomaly validation
    python scripts/validate_model.py individual --synthetic
    python scripts/validate_model.py cross_sectional --synthetic
        """
    )

    parser.add_argument('model_type', 
                        choices=['individual', 'cross_sectional'],
                        help='Model type to validate (individual or cross_sectional)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic anomaly validation instead of temporal backtest')
    parser.add_argument('--test-start', type=str, default=None,
                        help='Test period start date (temporal mode only, default: day after training ended)')
    parser.add_argument('--test-end', type=str, default=None,
                        help='Test period end date (temporal mode only, default: latest available)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top anomalies to show per date (temporal mode only, default: 10)')
    parser.add_argument('--n-anomalies', type=int, default=100,
                        help='Number of synthetic anomalies to create (synthetic mode only, default: 100)')

    args = parser.parse_args()

    if args.synthetic:
        validate_synthetic(
            args.model_type,
            n_anomalies=args.n_anomalies
        )
    else:
        validate_model(
            args.model_type,
            test_start=args.test_start,
            test_end=args.test_end,
            top_k=args.top_k
        )


if __name__ == '__main__':
    main()
