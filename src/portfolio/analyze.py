import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_portfolio_health(
    holdings,
    analysis_date,
    model_type,
    market_proxy="SPY",
    contra_horizon=5,
    db_path=None,
):
    """
    Analyzes the health of a given stock portfolio based on structural anomaly
    scores and directional market deviations.
    """

    # ------------------------------------------------------------------
    # Database connection
    # ------------------------------------------------------------------
    if db_path is None:
        db_path = Path("data/processed/market_data.sqlite")
    else:
        db_path = Path(db_path)

    conn = sqlite3.connect(db_path)

    try:
        # ------------------------------------------------------------------
        # Normalize portfolio weights
        # ------------------------------------------------------------------
        weights = {h["symbol"]: h["weight"] for h in holdings}
        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("Total portfolio weight must be > 0")

        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # ------------------------------------------------------------------
        # Load AE scores for analysis date
        # ------------------------------------------------------------------
        scored = pd.read_sql(
            """
            SELECT symbol, ae_score
            FROM stock_anomaly_scores
            WHERE date = ? AND model_type = ?
            """,
            conn,
            params=(analysis_date, model_type),
        )

        scored = scored[scored["symbol"].isin(norm_weights.keys())].copy()
        missing_symbols = sorted(set(norm_weights.keys()) - set(scored["symbol"]))

        if scored.empty:
            raise ValueError("No portfolio symbols have AE scores for this date")

        scored["weight"] = scored["symbol"].map(norm_weights)
        scored["contribution"] = scored["weight"] * scored["ae_score"]

        portfolio_score = float(scored["contribution"].sum())

        # ------------------------------------------------------------------
        # Market baseline for structural normalization
        # ------------------------------------------------------------------
        mkt = pd.read_sql(
            """
            SELECT ae_mean, ae_std, ae_p95, ae_p99, n
            FROM market_daily_summary
            WHERE date = ? AND model_type = ?
            """,
            conn,
            params=(analysis_date, model_type),
        )

        if mkt.empty:
            raise ValueError("Missing market baseline for analysis date")

        mkt = mkt.iloc[0]
        structural_z = (portfolio_score - mkt["ae_mean"]) / max(mkt["ae_std"], 1e-8)

        # ------------------------------------------------------------------
        # Contra return vs market proxy
        # ------------------------------------------------------------------
        start_date = (
            pd.to_datetime(analysis_date) - timedelta(days=contra_horizon * 3)
        ).strftime("%Y-%m-%d")

        symbols = [market_proxy] + scored["symbol"].tolist()

        prices = pd.read_sql(
            f"""
            SELECT date, symbol, close
            FROM market_prices
            WHERE symbol IN ({",".join("?" * len(symbols))})
              AND date BETWEEN ? AND ?
            """,
            conn,
            params=tuple(symbols + [start_date, analysis_date]),
        )

        if prices.empty:
            raise ValueError("No price data available for contra-return calculation")

        prices["date"] = prices["date"].astype(str)
        prices = prices.pivot(index="date", columns="symbol", values="close")

        # Compute returns independently per symbol
        returns = prices.pct_change(contra_horizon)

        # Drop dates with no return observations
        valid_dates = returns.dropna(how="all").index
        if len(valid_dates) == 0:
            raise ValueError("No valid return observations")

        latest_date = valid_dates.max()
        latest_returns = returns.loc[latest_date]

        # Market proxy return
        if market_proxy not in latest_returns or pd.isna(latest_returns[market_proxy]):
            raise ValueError("Missing market proxy return")

        market_ret = float(latest_returns[market_proxy])

        # Portfolio symbols with valid returns
        usable = scored["symbol"].isin(latest_returns.index) & ~latest_returns[
            scored["symbol"]
        ].isna().values

        used = scored.loc[usable].copy()

        if used.empty:
            raise ValueError("No portfolio symbols have valid return data")

        used["weight"] = used["weight"] / used["weight"].sum()

        portfolio_ret = float(
            np.dot(
                latest_returns[used["symbol"]].values,
                used["weight"].values,
            )
        )

        contra_return = portfolio_ret - market_ret

        market_vol = returns[market_proxy].std()
        directional_z = (-contra_return) / max(market_vol, 1e-8)

        health_score = float(structural_z + directional_z)

        result = {
            # Identity
            "date": analysis_date,
            "model_type": model_type,

            # Structural divergence (autoencoder-based)
            "structural_score": portfolio_score,

            "structural_z": float(structural_z),

            # Directional / contra movement
            "directional_score": float(contra_return),

            "directional_z": float(directional_z),

            # Combined portfolio health
            "health_score": float(health_score),

            # Market baseline context
            "market_baseline": {
                "ae_mean": float(mkt["ae_mean"]),
                "ae_std": float(mkt["ae_std"]),
                "ae_p95": float(mkt["ae_p95"]),
                "ae_p99": float(mkt["ae_p99"]),
                "n": int(mkt["n"]),
            },

            # Coverage / data integrity
            "coverage": {
                "n_requested": len(holdings),
                "n_scored": len(scored),
                "n_used_returns": len(used),
                "missing_symbols": missing_symbols,
                "weights_renormalized": (
                        abs(total_weight - 1.0) > 1e-6 or
                        len(used) < len(scored)
                ),
            },

            # Attribution - structural contribution
            "contributors": scored.sort_values(
                "contribution", ascending=False
            )[
                ["symbol", "weight", "ae_score", "contribution"]
            ].to_dict(orient="records"),
        }

        return result

    finally:
        conn.close()

