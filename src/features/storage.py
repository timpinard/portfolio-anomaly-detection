"""Feature storage utilities for experiment persistence."""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Manages persistent storage of experiment features.

    Features are stored in experiment-specific tables to allow
    different feature sets for different experiments.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    def _get_table_name(self, experiment_name: str) -> str:
        """Get table name for an experiment's features."""
        # Sanitize experiment name for SQL
        safe_name = experiment_name.replace('-', '_').replace(' ', '_')
        return f"features_{safe_name}"

    def table_exists(self, experiment_name: str) -> bool:
        """Check if features table exists for an experiment."""
        table_name = self._get_table_name(experiment_name)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        exists = cursor.fetchone() is not None
        conn.close()

        return exists

    def get_feature_count(self, experiment_name: str) -> int:
        """Get number of feature rows for an experiment."""
        if not self.table_exists(experiment_name):
            return 0

        table_name = self._get_table_name(experiment_name)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def save_features(
        self,
        experiment_name: str,
        features_df: pd.DataFrame,
        overwrite: bool = False
    ) -> None:
        """
        Save features to database.

        Args:
            experiment_name: Name of the experiment
            features_df: DataFrame with features
            overwrite: If True, replace existing table
        """
        table_name = self._get_table_name(experiment_name)

        conn = sqlite3.connect(self.db_path)

        if overwrite and self.table_exists(experiment_name):
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"Dropped existing table: {table_name}")

        df_to_save = features_df.copy()
        if isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save = df_to_save.reset_index()
            df_to_save.rename(columns={'index': 'date'}, inplace=True)

        for col in df_to_save.columns:
            if pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
                df_to_save[col] = df_to_save[col].astype(str)

        df_to_save.to_sql(table_name, conn, if_exists='replace', index=False)

        cursor = conn.cursor()
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
            if 'date' in df_to_save.columns:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
        except Exception as e:
            logger.warning(f"Could not create index: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Saved {len(features_df):,} feature rows to {table_name}")

    def load_features(
        self,
        experiment_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load features from database.

        Args:
            experiment_name: Name of the experiment
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with features
        """
        if not self.table_exists(experiment_name):
            logger.warning(f"No features table found for experiment: {experiment_name}")
            return pd.DataFrame()

        table_name = self._get_table_name(experiment_name)

        conn = sqlite3.connect(self.db_path)

        query = f"SELECT * FROM {table_name}"
        params = []

        conditions = []
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded {len(df):,} feature rows from {table_name}")

        return df

    def delete_features(self, experiment_name: str) -> bool:
        """Delete features table for an experiment."""
        if not self.table_exists(experiment_name):
            return False

        table_name = self._get_table_name(experiment_name)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        conn.close()

        logger.info(f"Deleted features table: {table_name}")
        return True

    def list_experiments(self) -> list:
        """List all experiments with stored features."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'features_%'"
        )
        tables = [row[0].replace('features_', '') for row in cursor.fetchall()]
        conn.close()

        return tables
