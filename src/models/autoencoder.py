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
        
        # Only initialize model if input_dim is provided (for training)
        if input_dim is not None:
            self.model = Autoencoder(input_dim, encoding_dim, hidden_dims).to(self.device)
        else:
            self.model = None
        
    def fit(self, X, epochs=100, batch_size=32, learning_rate=0.001):
        """Train autoencoder."""
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
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
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        
        # Calculate threshold (99th percentile of reconstruction error)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.threshold = torch.quantile(errors, 0.99).item()
    
    def predict(self, X):
        """Predict anomalies (1 = normal, -1 = anomaly)."""
        scores = self.score(X)
        return (scores > self.threshold).astype(int) * -2 + 1  # Convert to -1/1
    
    def score(self, X):
        """Return reconstruction error as anomaly score."""
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        return errors.cpu().numpy()
    
    def save(self):
        """Save PyTorch model artifacts."""
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model state
        torch.save(self.model.state_dict(), self.model_dir / 'model.pth')
        
        # Save scaler
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        
        # Save metadata including architecture and threshold
        metadata = {
            'sector': self.sector,
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'device': self.device
        }
        with open(self.model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {self.model_dir}")
    
    def load(self):
        """Load PyTorch model artifacts."""
        # Load metadata first to reconstruct model architecture
        with open(self.model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Set attributes from metadata
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_dims = metadata['hidden_dims']
        self.threshold = metadata.get('threshold')
        self.device = metadata.get('device', 'cpu')
        
        # Reconstruct model architecture
        self.model = Autoencoder(
            self.input_dim, 
            self.encoding_dim, 
            self.hidden_dims
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(
            torch.load(self.model_dir / 'model.pth', map_location=self.device)
        )
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        
        print(f"Model loaded from {self.model_dir}")
