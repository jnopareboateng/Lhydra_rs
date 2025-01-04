import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import os
from train_model import HybridMusicRecommender, MusicRecommenderDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add safe globals for numpy types
torch.serialization.add_safe_globals([
    np.generic,  # Allow numpy scalar types
    np.ndarray,  # Allow numpy arrays
    np.dtype,    # Allow numpy dtypes
    np.float64,  # Allow specific numpy types
    np.float32,
    np.int64,
    np.int32
])

class RecommendationGenerator:
    def __init__(self, model_path: str, catalog_data: pd.DataFrame, encoders_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.catalog_data = catalog_data
        
        # Load model checkpoint with safety settings
        logger.info(f"Loading model from {model_path}")
        try:
            self.checkpoint = torch.load(model_path, map_location=self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Get config and encoders from the checkpoint
        self.config = self.checkpoint.get('config', {})
        if not self.config:
            # Try loading from config file as fallback
            try:
                with open('config/model_config.json', 'r') as f:
                    self.config = json.load(f)
            except FileNotFoundError:
                logger.warning("Config file not found, using default values")
                self.config = {
                    'embedding_dim': 64,
                    'hidden_layers': [256, 128, 64],
                    'dropout': 0.3
                }
        
        # Load encoders with safety settings
        torch.serialization.add_safe_globals([LabelEncoder])
        self.encoders = torch.load(encoders_path, weights_only=False)
        
        # Print shape info for debugging
        logger.info("Encoder class counts:")
        for key, encoder in self.encoders.items():
            if isinstance(encoder, LabelEncoder):
                logger.info(f"{key}: {len(encoder.classes_)}")
        
        # Get state dict dimensions with safety checks
        state_dict = self.checkpoint['model_state_dict']
        self.embedding_dims = {
            'num_users': state_dict['user_embedding.weight'].shape[0],
            'num_music': state_dict['music_embedding.weight'].shape[0],
            'num_artists': state_dict['artist_embedding.weight'].shape[0],
            'num_genres': len(self.encoders['genre_encoder'].classes_),
            'num_numerical': 12
        }
        
        logger.info("Model dimensions from state dict:")
        for key, value in self.embedding_dims.items():
            logger.info(f"{key}: {value}")
        
        # Safety check for catalog data
        max_music_id = self.catalog_data['music_id'].nunique()
        if max_music_id >= self.embedding_dims['num_music']:
            logger.warning(f"Catalog contains music IDs larger than model capacity. Filtering out excess items.")
            valid_music_ids = set(self.encoders['music_encoder'].transform(
                self.encoders['music_encoder'].classes_[:self.embedding_dims['num_music']]
            ))
            self.catalog_data = self.catalog_data[
                self.catalog_data['music_id'].apply(
                    lambda x: self.encoders['music_encoder'].transform([x])[0] in valid_music_ids
                )
            ]
            logger.info(f"Filtered catalog size: {len(self.catalog_data)}")
        
        self.model = self._initialize_model(self.embedding_dims)
        
    def _initialize_model(self, embedding_dims):
        """Initialize and load the model from checkpoint."""
        # Get dimensions from encoders
        model = HybridMusicRecommender(
            num_users=embedding_dims['num_users'],
            num_music=embedding_dims['num_music'],
            num_artists=embedding_dims['num_artists'],
            num_genres=embedding_dims['num_genres'],
            num_numerical=embedding_dims['num_numerical'],
            embedding_dim=64,
            layers=[256, 128, 64],
            dropout=0.2
        )
        
        # Load state dict from checkpoint
        state_dict = self.checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        
        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def generate_recommendations(self, user_info: dict, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Generate music recommendations for a specific user.
        
        Args:
            user_info: Dictionary containing user information (age, gender, user_id)
            n_recommendations: Number of recommendations to generate
            
        Returns:
            DataFrame containing recommended songs with predicted play counts
        """
        # Create a temporary DataFrame with all songs for the user
        user_candidates = self.catalog_data.copy()
        user_candidates['age'] = user_info['age']
        user_candidates['gender'] = user_info['gender']
        user_candidates['user_id'] = user_info['user_id']
        
        # Debug user encoding with more detailed error handling
        try:
            encoded_user = self.encoders['user_encoder'].transform([user_info['user_id']])[0]
            logger.info(f"User ID {user_info['user_id']} encoded as: {encoded_user}")
        except Exception as e:
            logger.warning(f"Error encoding user ID: {str(e)}")
            logger.warning("Using default encoding (0)")
            encoded_user = 0
            user_candidates['user_id'] = '0'  # Use default user ID
        
        # Debug catalog data
        print(f"\nCatalog Statistics:")
        print(f"Total songs: {len(user_candidates)}")
        print(f"Unique artists: {user_candidates['artist_name'].nunique()}")
        print(f"Unique genres: {user_candidates['main_genre'].nunique()}")
        
        try:
            # Create dataset with safety checks
            test_dataset = MusicRecommenderDataset(
                user_candidates,
                mode='test',
                encoders=self.encoders,
                embedding_dims=self.embedding_dims  # Pass embedding dimensions
            )
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            # Generate predictions
            predictions = []
            indices = []
            
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    pred = self.model(batch)
                    predictions.extend(pred.cpu().numpy())
                    indices.extend(range(i * test_loader.batch_size, 
                                      min((i + 1) * test_loader.batch_size, len(test_dataset))))
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
        
        # Create recommendations DataFrame and ensure uniqueness
        recommendations = pd.DataFrame({
            'music': user_candidates['music'].values[indices],
            'artist_name': user_candidates['artist_name'].values[indices],
            'genre': user_candidates['main_genre'].values[indices],
            'predicted_plays': predictions
        })
        
        # Drop duplicates keeping first occurrence (highest predicted play count)
        recommendations = recommendations.drop_duplicates(subset=['music'], keep='first')
        
        # Convert predictions to scalar values and sort
        recommendations['predicted_plays'] = recommendations['predicted_plays'].apply(lambda x: float(x[0]))
        
        # Sort by predicted plays and get top N recommendations
        recommendations = recommendations.sort_values('predicted_plays', ascending=False)
        recommendations = recommendations.head(n_recommendations)
        
        # Debug predictions
        print(f"\nPrediction Statistics:")
        min_pred = recommendations['predicted_plays'].min()
        max_pred = recommendations['predicted_plays'].max()
        std_pred = recommendations['predicted_plays'].std()
        print(f"Prediction range: {min_pred:.2f} to {max_pred:.2f}")
        print(f"Prediction std: {std_pred:.2f}")
        
        # Print top recommendations with better formatting
        print("\nTop 10 Recommended Songs:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(recommendations.to_string(index=False, float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, (float, np.float32, np.float64)) else str(x)))
        
        return recommendations.reset_index(drop=True)

class HybridMusicRecommender(nn.Module):
    def __init__(self, num_users, num_music, num_artists, num_genres, num_numerical,
                 embedding_dim=64, layers=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        # Embedding layers
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
        total_features = embedding_dim * 6  # 4 embeddings + numerical + binary
        
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
    
    def forward(self, batch):
        # Get embeddings
        user_emb = self.user_embedding(batch['user_id'])
        music_emb = self.music_embedding(batch['music_id'])
        artist_emb = self.artist_embedding(batch['artist_id'])
        genre_emb = self.genre_embedding(batch['genre_id'])
        
        # Process numerical features
        numerical = self.numerical_layer(batch['numerical_features'])
        
        # Process binary features
        binary = torch.stack([batch['explicit'], batch['gender']], dim=1).float()
        binary = self.binary_layer(binary)
        
        # Concatenate all features
        x = torch.cat([
            user_emb, music_emb, artist_emb, genre_emb, numerical, binary
        ], dim=1)
        
        # Apply MLP layers with residual connections
        for layer in self.fc_layers:
            identity = x
            x = layer['main'](x)
            if layer['residual'] is not None:
                x = x + layer['residual'](identity)
        
        # Final prediction
        return self.final_layer(x)

def main():
    # Example usage
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = 'checkpoints/best_model.pth'
    catalog_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test_data.csv'))
    encoders_path = os.path.join(BASE_DIR, 'data', 'data_encoders.pt')
    
    # Initialize recommendation generator
    recommender = RecommendationGenerator(model_path, catalog_data, encoders_path)
    
    # Example user
    user_info = {
        'age': 32,
        'gender': 'M',
        'genre': 'Pop',
        'music': 'Shape of You',
        'user_id': '44d39c6e5e7b45bfc2187fb3c89be58c5a3dc6a54d2a0075402c551c14ea1459'
    }
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(user_info, n_recommendations=10)
    
    print("\nTop 10 Recommended Songs:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
