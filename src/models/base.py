from abc import ABC, abstractmethod
import joblib
from pathlib import Path

class BaseAnomalyModel(ABC):
    """Abstract base class for all anomaly detection models."""
    
    def __init__(self, sector: str, model_dir: Path):
        self.sector = sector
        self.model_dir = model_dir / sector
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    @abstractmethod
    def fit(self, X, y=None):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict anomalies."""
        pass
    
    @abstractmethod
    def score(self, X):
        """Return anomaly scores."""
        pass
    
    def save(self):
        """Save model artifacts."""
        joblib.dump(self.model, self.model_dir / 'model.joblib')
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
        joblib.dump(self.feature_names, self.model_dir / 'features.joblib')
    
    def load(self):
        """Load model artifacts."""
        self.model = joblib.load(self.model_dir / 'model.joblib')
        self.scaler = joblib.load(self.model_dir / 'scaler.joblib')
        self.feature_names = joblib.load(self.model_dir / 'features.joblib')
