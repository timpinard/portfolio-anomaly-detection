"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import sqlite3
import logging

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All feature extractors should inherit from this class and implement
    the required methods. This enables the factory pattern for experiments.
    """

    # Metadata about this extractor
    name: str = "base"
    description: str = "Base feature extractor"

    def __init__(self, db_path: str, config: Dict[str, Any] = None):
        """
        Initialize extractor.

        Args:
            db_path: Path to SQLite database with market_prices table
            config: Optional configuration dictionary for this extractor
        """
        self.db_path = Path(db_path)
        self.config = config or {}

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def load_price_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load price data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD) or None for all
            end_date: End date (YYYY-MM-DD) or None for all

        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM market_prices WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df.set_index('date', inplace=True)

        return df

    def get_all_symbols(self) -> List[str]:
        """Get all symbols in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols

    @abstractmethod
    def calculate_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate features for a single symbol's price data.

        Args:
            df: DataFrame with OHLCV data for one symbol
            **kwargs: Additional arguments (e.g., market data for cross-sectional)

        Returns:
            DataFrame with calculated features
        """
        pass

    @abstractmethod
    def prepare_training_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare training data for all symbols.

        Args:
            symbols: List of symbols (None = all symbols in database)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Combined DataFrame with features for all symbols
        """
        pass

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata).

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude_cols = {'symbol', 'date', 'index'}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def prepare_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for model input.

        Args:
            df: DataFrame with features

        Returns:
            Feature matrix (n_samples, n_features)
        """
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].values

        # Replace inf and NaN with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X
