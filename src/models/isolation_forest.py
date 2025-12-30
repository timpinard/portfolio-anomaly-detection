import joblib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base import BaseAnomalyModel

class IsolationForestAnomalyDetector(BaseAnomalyModel):
    """Isolation Forest-based anomaly detection."""
    
    def __init__(self, sector, model_dir, contamination=0.1, n_estimators=100, 
                 max_samples='auto', random_state=42):
        super().__init__(sector, model_dir)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        # Initialize model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )
        
    def fit(self, X, y=None):
        """Train isolation forest."""
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        self.scaler = StandardScaler() 
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
    
    def predict(self, X):
        """Predict anomalies (1 = normal, -1 = anomaly)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X):
        """Return anomaly scores (lower = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        # Negate decision function so higher = more anomalous (consistent with autoencoder)
        return -self.model.decision_function(X_scaled)
    
    def save(self):
        """Save model artifacts."""
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.model_dir / 'model.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
        
        # Save feature names
        joblib.dump(self.feature_names, self.model_dir / 'features.joblib')
        
        # Save metadata
        metadata = {
            'sector': self.sector,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'random_state': self.random_state
        }
        with open(self.model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {self.model_dir}")
    
    def load(self):
        """Load model artifacts."""
        # Load model
        self.model = joblib.load(self.model_dir / 'model.joblib')
        
        # Load scaler
        self.scaler = joblib.load(self.model_dir / 'scaler.joblib')
        
        # Load feature names
        self.feature_names = joblib.load(self.model_dir / 'features.joblib')
        
        # Load metadata
        with open(self.model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Set attributes from metadata
        self.contamination = metadata.get('contamination', 0.1)
        self.n_estimators = metadata.get('n_estimators', 100)
        self.max_samples = metadata.get('max_samples', 'auto')
        self.random_state = metadata.get('random_state', 42)
        
        print(f"Model loaded from {self.model_dir}")
