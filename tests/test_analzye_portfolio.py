import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio.analyze import analyze_portfolio


def _create_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE stock_anomaly_scores (
            date       TEXT,
            symbol     TEXT,
            ae_score   REAL,
            ae_score_z REAL,
            model_type TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE market_prices (
            date   TEXT,
            symbol TEXT,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE market_daily_summary (
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

    conn.commit()
    conn.close()


def _insert_scores(db_path: Path, rows):
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(rows)
    df.to_sql("stock_anomaly_scores", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def _insert_prices(db_path: Path, rows):
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(rows)
    df.to_sql("market_prices", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def _insert_market_summary(db_path: Path, rows):
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame(rows)
    df.to_sql("market_daily_summary", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


@pytest.fixture
def temp_db(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    _create_db(db_path)
    return db_path


def test_analyze_portfolio_portfolio_score_and_contributors(temp_db: Path):
    # Given
    analysis_date = "2024-01-10"
    model_type = "cross_sectional"

    holdings = [
        {"symbol": "AAA", "weight": 2.0},  # intentionally not normalized
        {"symbol": "BBB", "weight": 1.0},
    ]

    _insert_scores(temp_db, [
        {"date": analysis_date, "symbol": "AAA", "ae_score": 0.30, "ae_score_z": 1.0, "model_type": model_type, "created_at": "x"},
        {"date": analysis_date, "symbol": "BBB", "ae_score": 0.10, "ae_score_z": 0.0, "model_type": model_type, "created_at": "x"},
    ])

    _insert_market_summary(temp_db, [
        {
            "date": analysis_date,
            "model_type": model_type,
            "n": 100,
            "ae_mean": 0.15,
            "ae_std": 0.05,
            "ae_p50": 0.14,
            "ae_p95": 0.25,
            "ae_p99": 0.30,
            "created_at": "x",
        }
    ])

    # Prices: need at least (analysis_date - horizon) and analysis_date for each symbol + SPY
    # Use contra_horizon=1 for simpler expected math
    # On 2024-01-09 -> 2024-01-10:
    # AAA: 100 -> 90  => -10%
    # BBB: 100 -> 100 => 0%
    # SPY: 100 -> 105 => +5%
    # Portfolio return (wAAA=2/3, wBBB=1/3): (2/3 * -10%) + (1/3 * 0%) = -6.6667%
    # Contra = -6.6667% - 5% = -11.6667% = -0.116667
    _insert_prices(temp_db, [
        {"date": "2024-01-09", "symbol": "AAA", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},
        {"date": "2024-01-10", "symbol": "AAA", "open": 0, "high": 0, "low": 0, "close": 90,  "volume": 0},

        {"date": "2024-01-09", "symbol": "BBB", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},
        {"date": "2024-01-10", "symbol": "BBB", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},

        {"date": "2024-01-09", "symbol": "SPY", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},
        {"date": "2024-01-10", "symbol": "SPY", "open": 0, "high": 0, "low": 0, "close": 105, "volume": 0},
    ])

    # When
    result = analyze_portfolio(
        analysis_date=analysis_date,
        holdings=holdings,
        db_path=temp_db,
        model_type=model_type,
        market_proxy="SPY",
        contra_horizon=1,
    )

    # Then: portfolio_score = (2/3)*0.30 + (1/3)*0.10 = 0.233333...
    assert np.isclose(result["portfolio_score"], (2/3)*0.30 + (1/3)*0.10, atol=1e-8)

    contributors = result["contributors"]
    assert contributors[0]["symbol"] == "AAA"
    assert np.isclose(contributors[0]["weight"], 2/3, atol=1e-8)
    assert np.isclose(contributors[0]["contribution"], (2/3)*0.30, atol=1e-8)

    assert np.isclose(result["contra_return"], -0.1166666667, atol=1e-6)


def test_analyze_portfolio_missing_scores_raises(temp_db: Path):
    analysis_date = "2024-01-10"
    model_type = "cross_sectional"

    holdings = [{"symbol": "AAA", "weight": 1.0}]

    _insert_prices(temp_db, [
        {"date": "2024-01-09", "symbol": "AAA", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},
        {"date": "2024-01-10", "symbol": "AAA", "open": 0, "high": 0, "low": 0, "close": 90,  "volume": 0},
        {"date": "2024-01-09", "symbol": "SPY", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},
        {"date": "2024-01-10", "symbol": "SPY", "open": 0, "high": 0, "low": 0, "close": 105, "volume": 0},
    ])

    with pytest.raises(ValueError, match="No portfolio symbols have AE scores for this date"):
        analyze_portfolio(
            analysis_date=analysis_date,
            holdings=holdings,
            db_path=temp_db,
            model_type=model_type,
            market_proxy="SPY",
            contra_horizon=1,
        )


def test_analyze_portfolio_missing_market_proxy_prices_raises(temp_db: Path):
    analysis_date = "2024-01-10"
    model_type = "cross_sectional"

    holdings = [{"symbol": "AAA", "weight": 1.0}]

    _insert_scores(temp_db, [
        {"date": analysis_date, "symbol": "AAA", "ae_score": 0.2, "ae_score_z": 0.0, "model_type": model_type, "created_at": "x"},
    ])

    _insert_market_summary(temp_db, [
        {
            "date": analysis_date,
            "model_type": model_type,
            "n": 100,
            "ae_mean": 0.15,
            "ae_std": 0.05,
            "ae_p50": 0.14,
            "ae_p95": 0.25,
            "ae_p99": 0.30,
            "created_at": "x",
        }
    ])

    _insert_prices(temp_db, [
        {"date": "2024-01-09", "symbol": "AAA", "open": 0, "high": 0, "low": 0, "close": 100, "volume": 0},
        {"date": "2024-01-10", "symbol": "AAA", "open": 0, "high": 0, "low": 0, "close": 90,  "volume": 0},
    ])

    with pytest.raises(ValueError, match="Missing market proxy return"):
        analyze_portfolio(
            analysis_date=analysis_date,
            holdings=holdings,
            db_path=temp_db,
            model_type=model_type,
            market_proxy="SPY",
            contra_horizon=1,
        )
