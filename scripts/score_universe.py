#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.autoencoder import AutoencoderAnomalyDetector
from features.storage import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_market_daily_summary(conn, scored_df, model_type):
    """
    Persist daily market-level AE score statistics.
    """
    scored_df = scored_df.copy()
    if pd.api.types.is_datetime64_any_dtype(scored_df["date"]):
        scored_df["date"] = scored_df["date"].dt.strftime("%Y-%m-%d")
    else:
        scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.strftime("%Y-%m-%d")

    g = scored_df.groupby("date")["ae_score"]

    summary = g.agg(["count", "mean", "std"]).reset_index()
    summary.rename(
        columns={
            "count": "n",
            "mean": "ae_mean",
            "std": "ae_std",
        },
        inplace=True,
    )

    q50 = g.quantile(0.50).reset_index(name="ae_p50")
    q95 = g.quantile(0.95).reset_index(name="ae_p95")
    q99 = g.quantile(0.99).reset_index(name="ae_p99")

    summary = summary.merge(q50, on="date").merge(q95, on="date").merge(q99, on="date")

    summary["model_type"] = model_type
    summary["created_at"] = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_daily_summary (
            date TEXT NOT NULL,
            model_type TEXT NOT NULL,
            n INTEGER NOT NULL,
            ae_mean REAL NOT NULL,
            ae_std REAL NOT NULL,
            ae_p50 REAL NOT NULL,
            ae_p95 REAL NOT NULL,
            ae_p99 REAL NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (date, model_type)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mds_model_type_date
        ON market_daily_summary (model_type, date)
    """)

    conn.executemany(
        "DELETE FROM market_daily_summary WHERE date = ? AND model_type = ?",
        [(d, model_type) for d in summary["date"].tolist()]  # now strings
    )

    summary.to_sql("market_daily_summary", conn, if_exists="append", index=False)
    conn.commit()

    logger.info(f"Saved market_daily_summary ({len(summary)} days)")


def score_universe(model_type):
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "processed" / "market_data.sqlite"

    exp_dir = project_root / "models" / "model_types" / model_type
    ae_dir = exp_dir / "autoencoder"

    conn = sqlite3.connect(db_path)

    logger.info("Loading Autoencoder and scaler")
    ae = AutoencoderAnomalyDetector.load(ae_dir)

    scaler = joblib.load(exp_dir / "scaler.joblib")
    feature_cols = joblib.load(exp_dir / "feature_columns.joblib")

    store = FeatureStore(str(db_path))
    features = store.load_features(model_type)

    X = features[feature_cols].values
    X = np.nan_to_num(X)
    X = scaler.transform(X)

    logger.info("Scoring universe")
    scores = ae.score(X)

    out = features[["date", "symbol"]].copy()
    out["ae_score"] = scores
    out["model_type"] = model_type
    out["created_at"] = datetime.now(timezone.utc).isoformat()
    
    if pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    else:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    out.to_sql("stock_anomaly_scores", conn, if_exists="append", index=False)
    logger.info(f"Saved {len(out):,} anomaly scores")

    write_market_daily_summary(conn, out[["date", "ae_score"]], model_type)

    conn.close()
    logger.info("Universe scoring complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    args = parser.parse_args()
    score_universe(args.model_type)


if __name__ == "__main__":
    main()
