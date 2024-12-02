import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import logging
import wandb
from tqdm import tqdm
import os
from typing import Tuple, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRecommenderDataset(Dataset):
    """Custom Dataset for loading music recommendation data with additional features."""
    
    def __init__(self, df: pd.DataFrame, mode: str = 'train'):
        self.mode = mode
        
        # Create encoders for categorical variables
        self.user_encoder = LabelEncoder()
        self.music_encoder = LabelEncoder()
        self.artist_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        
        # Fit encoders
        self.users = self.user_encoder.fit_transform(df['user_id'].values)
        self.music = self.music_encoder.fit_transform(df['music_id'].values)
        self.artists = self.artist_encoder.fit_transform(df['artist_id'].values)
        self.genres = self.genre_encoder.fit_transform(df['genre'].values)
        
        # Scale numerical features
        self.scaler = RobustScaler()
        numerical_features = [
            'age', 'duration', 'acousticness', 'key', 'mode', 'speechiness',
            'instrumentalness', 'liveness', 'tempo', 'time_signature',
            'music_age', 'plays', 'energy_loudness', 'dance_valence'
        ]
        
        self.numerical_features = self.scaler.fit_transform(df[numerical_features].values)
        
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
            'plays': torch.tensor(self.numerical_features[idx][11], dtype=torch.float)  # Target variable
        }

class HybridMusicRecommender(nn.Module):
    """Hybrid Neural Collaborative Filtering model with additional features."""
    
    def __init__(self, num_users: int, num_music: int, num_artists: int, 
                 num_genres: int, num_numerical: int, embedding_dim: int = 64,
                 layers: List[int] = [256, 128, 64], dropout: float = 0.2):
        super(HybridMusicRecommender, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.music_embedding = nn.Embedding(num_music, embedding_dim)
        self.artist_embedding = nn.Embedding(num_artists, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        
        # Feature processing layers
        self.numerical_layer = nn.Linear(num_numerical, embedding_dim)
        self.binary_layer = nn.Linear(2, embedding_dim)  # For explicit and gender
        
        # Calculate total input features
        total_features = embedding_dim * 6  # 4 embeddings + numerical + binary
        
        # MLP layers
        self.fc_layers = []
        input_dim = total_features
        
        for layer_size in layers:
            self.fc_layers.extend([
                nn.Linear(input_dim, layer_size),
                nn.ReLU(),
                nn.BatchNorm1d(layer_size),
                nn.Dropout(dropout)
            ])
            input_dim = layer_size
        
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.final_layer = nn.Linear(layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
                
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        combined = torch.cat([
            user_emb, music_emb, artist_emb, genre_emb, 
            numerical_features, binary_features
        ], dim=1)
        
        # Pass through MLP layers
        x = self.fc_layers(combined)
        output = self.final_layer(x)
        
        return output.squeeze()

class Trainer:
    """Trainer class for the hybrid music recommender model."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            prediction = self.model(batch)
            loss = self.criterion(prediction, batch['plays'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                prediction = self.model(batch)
                loss = self.criterion(prediction, batch['plays'])
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self):
        """Train the model."""
        logger.info(f'Training on device: {self.device}')
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            logger.info(f'Epoch {epoch+1}/{self.config["epochs"]}')
            logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                         os.path.join(self.config['model_dir'], 'best_model.pth'))
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience']:
                logger.info('Early stopping triggered')
                break

def main():
    # Configuration
    config = {
        'batch_size': 512,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 10,
        'embedding_dim': 64,
        'model_dir': 'models',
        'hidden_layers': [256, 128, 64],
        'dropout': 0.3
    }
    
    # Initialize wandb
    wandb.init(project='music-recommender', config=config)

    # Load data
    df = pd.read_csv('/home/josh/Lhydra_rs/data_engineered/rs_main_v2_refactored/data/engineered_data.csv')
    
    # Create train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = MusicRecommenderDataset(train_df)
    val_dataset = MusicRecommenderDataset(val_df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    
    # Initialize model
    model = HybridMusicRecommender(
        num_users=len(train_dataset.user_encoder.classes_),
        num_music=len(train_dataset.music_encoder.classes_),
        num_artists=len(train_dataset.artist_encoder.classes_),
        num_genres=len(train_dataset.genre_encoder.classes_),
        num_numerical=14,  # Update this to match the actual number of numerical features
        embedding_dim=config['embedding_dim'],
        layers=config['hidden_layers'],
        dropout=config['dropout']
    )
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    main()
