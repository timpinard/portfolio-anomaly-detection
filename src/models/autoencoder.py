import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path
from .base import BaseAnomalyModel

class Autoencoder(nn.Module):
    """Autoencoder neural network for feature learning."""
    
    def __init__(self, input_dim, encoding_dim, hidden_dims=[64, 32]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoencoderAnomalyDetector(BaseAnomalyModel):
    """Autoencoder-based anomaly detection."""
    
    def __init__(self, sector, model_dir, input_dim=None, encoding_dim=10, 
                 hidden_dims=[64, 32], device='cuda'):
        super().__init__(sector, model_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.threshold = None
        self.history = {'loss': [], 'val_loss': []}  # Track training history
        
        # Only initialize model if input_dim is provided (for training)
        if input_dim is not None:
            self.model = Autoencoder(input_dim, encoding_dim, hidden_dims).to(self.device)
        else:
            self.model = None
        
    def fit(self, X, epochs=100, batch_size=32, learning_rate=0.001, threshold_percentile=99):
        """Train autoencoder.

        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            threshold_percentile: Percentile of reconstruction error to use as threshold (1-99).
                                  Higher = fewer false positives, lower recall.
                                  Lower = more false positives, higher recall.
        """
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Reset history
        self.history = {'loss': [], 'val_loss': []}
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in loader:
                data = batch[0].to(self.device)
                
                # Forward pass
                reconstructed = self.model(data)
                loss = criterion(reconstructed, data)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record average loss for this epoch
            avg_loss = epoch_loss / len(loader)
            self.history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Calculate threshold (99th percentile of reconstruction error)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            quantile = threshold_percentile / 100.0
            self.threshold = torch.quantile(errors, quantile).item()
    
    def predict(self, X):
        """Predict anomalies (1 = normal, -1 = anomaly)."""
        scores = self.score(X)
        return (scores > self.threshold).astype(int) * -2 + 1  # Convert to -1/1
    
    def score(self, X):
        """
        Compute reconstruction error for already-scaled inputs.
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self.model(X_tensor)

            # Mean squared reconstruction error per sample
            errors = torch.mean((X_tensor - recon) ** 2, dim=1)

        return errors.cpu().numpy()
    
    def save(self):
        import json
        from pathlib import Path

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), self.model_dir / "model.pt")

        # Save metadata
        metadata = {
            "sector": self.sector,
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dims": self.hidden_dims,
            "device": str(self.device),
            "threshold": self.threshold,
        }

        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)

        # --- load metadata ---
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {model_dir}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # --- reconstruct model ---
        model = cls(
            sector=metadata["sector"],
            model_dir=model_dir,
            input_dim=metadata["input_dim"],
            encoding_dim=metadata["encoding_dim"],
            hidden_dims=metadata["hidden_dims"],
            device=metadata.get("device", "cpu"),
        )

        # --- load weights ---
        weights_path = model_dir / "model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing model.pt in {model_dir}")

        model.model.load_state_dict(
            torch.load(weights_path, map_location=model.device)
        )

        model.threshold = metadata.get("threshold")

        return model