import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
import logging
from tqdm import tqdm
import os
from typing import Tuple, Dict, List
import json
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from encoder_utils import DataEncoder  # Add this import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MusicRecommenderDataset(Dataset):
    """Custom Dataset for loading music recommendation data with additional features."""
    
    def __init__(self, df: pd.DataFrame, mode: str = 'train', encoders=None, embedding_dims=None):
        self.df = df
        self.mode = mode
        self.embedding_dims = embedding_dims
        
        if encoders is not None:
            self.user_encoder = encoders['user_encoder']
            self.music_encoder = encoders['music_encoder']
            self.artist_encoder = encoders['artist_encoder']
            self.genre_encoder = encoders['genre_encoder']
            self.scaler = encoders['scaler']
            
            # Handle unknown values for each encoder
            def safe_transform(encoder, values, max_index=None, default_value=0):
                try:
                    transformed = encoder.transform(values)
                    if max_index is not None:
                        # Clip values to be within embedding range
                        transformed = np.clip(transformed, 0, max_index - 1)
                    logger.debug(f"Transformed shape: {transformed.shape}")
                    return transformed
                except Exception as e:
                    logger.warning(f"Error in transform: {str(e)}")
                    logger.warning(f"Using default value {default_value} for {len(values)} items")
                    return np.array([default_value] * len(values))
            
            # Transform with dimension limits
            max_dims = embedding_dims if embedding_dims else {}
            self.users = safe_transform(self.user_encoder, df['user_id'].values, 
                                      max_index=max_dims.get('num_users', None))
            self.music = safe_transform(self.music_encoder, df['music_id'].values, 
                                      max_index=max_dims.get('num_music', None))
            self.artists = safe_transform(self.artist_encoder, df['artist_id'].values, 
                                        max_index=max_dims.get('num_artists', None))
            self.genres = safe_transform(self.genre_encoder, df['main_genre'].values, 
                                       max_index=max_dims.get('num_genres', None))
            
            numerical_features = [
                'age', 'duration', 'acousticness', 'key', 'mode', 'speechiness',
                'instrumentalness', 'liveness', 'tempo', 'time_signature',
                'energy_loudness', 'dance_valence'
            ]
            
            # Handle numerical features
            try:
                self.numerical_features = self.scaler.transform(df[numerical_features].values)
            except KeyError as e:
                logger.warning(f"Missing numerical features: {str(e)}")
                self.numerical_features = np.zeros((len(df), len(numerical_features)))
            
            # Fix this part - Currently using numerical_features[11] which isn't playcount
            # Instead, use the actual playcount column
            self.playcount = df['playcount'].values  # Add this line
        else:
            raise ValueError("Encoders must be provided")

        # Binary features
        self.explicit = df['explicit'].astype(int).values
        self.gender = (df['gender'] == 'M').astype(int).values
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_music = len(self.music_encoder.classes_)
        self.num_artists = len(self.artist_encoder.classes_)
        self.num_genres = len(self.genre_encoder.classes_)
        self.num_numerical = len(numerical_features)
        
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'user_id': torch.tensor(self.users[idx], dtype=torch.long),
            'music_id': torch.tensor(self.music[idx], dtype=torch.long),
            'artist_id': torch.tensor(self.artists[idx], dtype=torch.long),
            'genre_id': torch.tensor(self.genres[idx], dtype=torch.long),
            'numerical_features': torch.tensor(self.numerical_features[idx], dtype=torch.float),
            'explicit': torch.tensor(self.explicit[idx], dtype=torch.float),
            'gender': torch.tensor(self.gender[idx], dtype=torch.float),
            'playcount': torch.tensor(self.playcount[idx], dtype=torch.float)  # Fix this line
        }

class HybridMusicRecommender(nn.Module):
    """Hybrid Neural Collaborative Filtering model with additional features."""
    
    def __init__(self, num_users: int, num_music: int, num_artists: int, 
                 num_genres: int, num_numerical: int, embedding_dim: int = 64,
                 layers: List[int] = [256, 128, 64], dropout: float = 0.2):
        super(HybridMusicRecommender, self).__init__()
        
        # Embedding layers with proper initialization
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.music_embedding = nn.Embedding(num_music, embedding_dim)
        self.artist_embedding = nn.Embedding(num_artists, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        
        # Feature processing layers with residual connections
        self.numerical_layer = nn.Sequential(
            nn.Linear(num_numerical, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        self.binary_layer = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Calculate total input features
        total_features = embedding_dim * 6
        
        # MLP layers with residual connections
        self.fc_layers = nn.ModuleList()
        input_dim = total_features
        
        for layer_size in layers:
            self.fc_layers.append(nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(input_dim, layer_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout)
                ),
                'residual': nn.Linear(input_dim, layer_size) if input_dim != layer_size else None
            }))
            input_dim = layer_size
        
        self.final_layer = nn.Linear(layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Input validation
        required_keys = ['user_id', 'music_id', 'artist_id', 'genre_id', 'numerical_features', 'explicit', 'gender']
        if not all(key in batch for key in required_keys):
            raise ValueError(f"Missing required keys in batch. Required: {required_keys}")
            
        # Get embeddings
        user_emb = self.user_embedding(batch['user_id'])
        music_emb = self.music_embedding(batch['music_id'])
        artist_emb = self.artist_embedding(batch['artist_id'])
        genre_emb = self.genre_embedding(batch['genre_id'])
        
        # Process numerical and binary features
        numerical_features = self.numerical_layer(batch['numerical_features'])
        binary_features = self.binary_layer(
            torch.stack([batch['explicit'], batch['gender']], dim=1)
        )
        
        # Combine all features
        x = torch.cat([
            user_emb, music_emb, artist_emb, genre_emb, 
            numerical_features, binary_features
        ], dim=1)
        
        # Pass through MLP layers with residual connections
        for layer in self.fc_layers:
            identity = x
            x = layer['main'](x)
            if layer['residual'] is not None:
                x = x + layer['residual'](identity)
            x = F.relu(x)
        
        return self.final_layer(x).squeeze()

def calculate_ndcg(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Calculate NDCG@K for rating predictions.
    For rating predictions, we consider higher predicted ratings as more relevant.
    """
    # Ensure inputs are on the same device
    device = predictions.device
    predictions = predictions.view(-1)  # Flatten predictions
    targets = targets.view(-1)  # Flatten targets
    
    # Sort predictions descending to get top K items
    _, indices = torch.sort(predictions, descending=True)
    indices = indices[:k]  # Get top K indices
    
    # Get corresponding target values
    pred_sorted = predictions[indices]
    target_sorted = targets[indices]
    
    # Calculate DCG
    pos = torch.arange(1, len(indices) + 1, device=device, dtype=torch.float32)
    dcg = (target_sorted / torch.log2(pos + 1)).sum()
    
    # Calculate IDCG
    ideal_target, _ = torch.sort(targets, descending=True)
    ideal_target = ideal_target[:k]
    idcg = (ideal_target / torch.log2(pos + 1)).sum()
    
    # Calculate NDCG, handling division by zero
    ndcg = dcg / (idcg + 1e-8)  # Add small epsilon to avoid division by zero
    return ndcg.item()

class Trainer:
    """Trainer class for the hybrid music recommender model."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Dict, encoders):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.encoders = encoders
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        self.model = self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Early stopping configuration
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Create directories for metrics and checkpoints
        os.makedirs('metrics', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        self.metrics_file = os.path.join('metrics', f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.metrics_history = {
            'train_loss': [], 'train_rmse': [], 'train_mae': [], 'train_ndcg': [],
            'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_ndcg': [],
            'lr': []
        }
        
        # Add L1 regularization
        self.l1_lambda = config.get('l1_lambda', 1e-5)
        
    def calculate_l1_loss(self, model):
        """Calculate L1 regularization loss."""
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss
        
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate training metrics."""
        # Convert tensors to numpy for sklearn metrics
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Calculate basic metrics
        mse = mean_squared_error(targets, predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(targets)
        
        # Calculate NDCG using tensor inputs
        ndcg = calculate_ndcg(
            torch.tensor(predictions, device=self.device),
            torch.tensor(targets, device=self.device),
            k=10
        )
        
        return {
            'loss': mse,
            'rmse': rmse,
            'mae': mae,
            'ndcg': ndcg
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_metrics = {'loss': 0.0, 'rmse': 0.0, 'mae': 0.0, 'ndcg': 0.0}
        num_batches = len(self.train_loader)
        
        for batch in tqdm(self.train_loader, desc='Training'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            # Add L1 regularization to loss
            loss = self.criterion(predictions, batch['playcount'])
            l1_loss = self.calculate_l1_loss(self.model)
            total_loss = loss + l1_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Calculate metrics
            batch_metrics = self.calculate_metrics(predictions.detach(), batch['playcount'])
            for k, v in batch_metrics.items():
                total_metrics[k] += v
                
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
        
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_metrics = {'loss': 0.0, 'rmse': 0.0, 'mae': 0.0, 'ndcg': 0.0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = self.model(batch)
                
                # Calculate metrics
                batch_metrics = self.calculate_metrics(predictions, batch['playcount'])
                for k, v in batch_metrics.items():
                    total_metrics[k] += v
                    
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'encoders': self.encoders
        }
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        # Save latest checkpoint
        checkpoint_path = os.path.join('checkpoints', 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if current model is best
        if is_best:
            best_model_path = os.path.join('checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
            
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            logger.info(f"Training metrics: {train_metrics}")
            
            # Validation
            val_metrics = self.validate()
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update metrics history
            current_lr = float(self.optimizer.param_groups[0]['lr'])  # Convert to Python float
            self.metrics_history['train_loss'].append(float(train_metrics['loss']))
            self.metrics_history['train_rmse'].append(float(train_metrics['rmse']))
            self.metrics_history['train_mae'].append(float(train_metrics['mae']))
            self.metrics_history['train_ndcg'].append(float(train_metrics['ndcg']))
            self.metrics_history['val_loss'].append(float(val_metrics['loss']))
            self.metrics_history['val_rmse'].append(float(val_metrics['rmse']))
            self.metrics_history['val_mae'].append(float(val_metrics['mae']))
            self.metrics_history['val_ndcg'].append(float(val_metrics['ndcg']))
            self.metrics_history['lr'].append(current_lr)
            
            # Save metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            
            # Check if current model is best
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            logger.info("-" * 50)

def cross_validate(data: pd.DataFrame, config: Dict, n_splits: int = 5):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    best_model_state = None
    best_val_score = float('inf')
    
    # Print CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Initialize encoders once on full dataset
    encoder = DataEncoder()
    encoder.fit(data)
    encoders = encoder.get_encoders()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        logger.info(f"Training fold {fold + 1}/{n_splits}")
        
        # Create datasets for this fold
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # Use the same encoders for both train and validation
        train_dataset = MusicRecommenderDataset(train_data, encoders=encoders)
        val_dataset = MusicRecommenderDataset(val_data, encoders=encoders)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        
        # Initialize model with dimensions from the common encoder
        model = HybridMusicRecommender(
            num_users=len(encoders['user_encoder'].classes_),
            num_music=len(encoders['music_encoder'].classes_),
            num_artists=len(encoders['artist_encoder'].classes_),
            num_genres=len(encoders['genre_encoder'].classes_),
            num_numerical=12,
            embedding_dim=config['embedding_dim'],
            layers=config['hidden_layers'],
            dropout=config['dropout']
        )
        
        # Train model
        trainer = Trainer(model, train_loader, val_loader, config, encoders)
        trainer.train(config['epochs'])
        
        # Get final validation metrics
        val_metrics = trainer.validate()
        fold_metrics.append(trainer.metrics_history)
        
        # Update best model if this fold performed better
        if val_metrics['loss'] < best_val_score:
            best_val_score = val_metrics['loss']
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'encoders': encoders,
                'fold': fold + 1,
                'metrics': val_metrics
            }
            logger.info(f"New best model from fold {fold + 1}")
    
    # Save the best model across all folds
    if best_model_state is not None:
        best_model_path = os.path.join('checkpoints', 'best_model_cv.pth')
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(best_model_state, best_model_path)
        logger.info(f"Saved best model from fold {best_model_state['fold']} to {best_model_path}")
    
    return fold_metrics

def main():
    # Configuration
    config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 20,
        'batch_size': 32,
        'embedding_dim': 64,
        'model_dir': 'models',
        'hidden_layers': [256, 128, 64],
        'dropout': 0.3,
        'early_stopping_patience': 2,
        'max_grad_norm': 1.0,
        'l1_lambda': 1e-5,  # L1 regularization strength
        'n_splits': 5,      # Number of cross-validation folds
    }
    
    # Save configuration
    os.makedirs('config', exist_ok=True)
    with open('config/model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load data and encoders
    train_data = pd.read_csv('../../data/train_data.csv')
    
    # Don't load existing encoders for cross-validation
    # Instead, let cross_validate create new encoders on the full dataset
    
    # Perform cross-validation
    fold_metrics = cross_validate(train_data, config)
    
    # Average metrics across folds
    avg_metrics = {
        'val_rmse': np.mean([m['val_rmse'][-1] for m in fold_metrics]),
        'val_ndcg': np.mean([m['val_ndcg'][-1] for m in fold_metrics])
    }
    
    logger.info(f"Cross-validation results:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
