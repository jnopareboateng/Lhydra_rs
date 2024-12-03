import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from train_v2 import HybridMusicRecommender, MusicRecommenderDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add safe globals for numpy types
torch.serialization.add_safe_globals([
    np._core.multiarray.scalar,  # Allow numpy scalar types
    np.ndarray,  # Allow numpy arrays
    np.dtype,    # Allow numpy dtypes
    np.float64,  # Allow specific numpy types
    np.float32,
    np.int64,
    np.int32
])

class RecommendationGenerator:
    def __init__(self, model_path: str, catalog_data: pd.DataFrame):
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
                with open('/home/josh/Lhydra_rs/data_engineered_v3/config/model_config.json', 'r') as f:
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
        self.encoders = torch.load('/home/josh/Lhydra_rs/data_engineered_v3/rs_main_v2_refactored/data/data_encoders.pt', weights_only=False)
        
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize and load the model from checkpoint."""
        # Get dimensions from encoders
        model = HybridMusicRecommender(
            num_users=len(self.encoders['user_encoder'].classes_),
            num_music=len(self.encoders['music_encoder'].classes_),
            num_artists=len(self.encoders['artist_encoder'].classes_),
            num_genres=len(self.encoders['genre_encoder'].classes_),
            num_numerical=14,  # Number of numerical features
            embedding_dim=64,  # Match the saved model's embedding dimension
            layers=[256, 128, 64],  # Match the saved model's layer sizes
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
        
        # Debug user encoding
        try:
            encoded_user = self.encoders['user_encoder'].transform([user_info['user_id']])[0]
            print(f"User ID {user_info['user_id']} encoded as: {encoded_user}")
        except:
            print(f"Warning: User ID {user_info['user_id']} not found in encoder")
            # Use a default encoding or handle unknown users
            encoded_user = 0
        
        # Debug catalog data
        print(f"\nCatalog Statistics:")
        print(f"Total songs: {len(user_candidates)}")
        print(f"Unique artists: {user_candidates['artist_name'].nunique()}")
        print(f"Unique genres: {user_candidates['genre'].nunique()}")
        
        # Create dataset and dataloader
        test_dataset = MusicRecommenderDataset(
            user_candidates,
            mode='test',
            encoders=self.encoders
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
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'music': user_candidates['music'].values[indices],
            'artist_name': user_candidates['artist_name'].values[indices],
            'genre': user_candidates['genre'].values[indices],
            'predicted_plays': predictions
        })
        
        # Convert predictions to scalar values
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
    model_path =  '/home/josh/Lhydra_rs/data_engineered_v3/checkpoints/best_model.pth'
    catalog_data = pd.read_csv('/home/josh/Lhydra_rs/data_engineered_v3/rs_main_v2_refactored/data/test_data.csv')  # Your music catalog
    
    # Initialize recommendation generator
    recommender = RecommendationGenerator(model_path, catalog_data)
    
    # Example user
    user_info = {
        'age': 32,
        'gender': 'M',
        'user_id': '44d39c6e5e7b45bfc2187fb3c89be58c5a3dc6a54d2a0075402c551c14ea1459'
    }
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(user_info, n_recommendations=10)
    
    print("\nTop 10 Recommended Songs:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
