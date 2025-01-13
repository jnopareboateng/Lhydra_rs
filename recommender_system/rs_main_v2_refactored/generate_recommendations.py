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
        try:
            encoder_dict = torch.load(encoders_path, weights_only=False)
            self.encoders = encoder_dict.get('encoder', None)  # This is now a DataEncoder object directly
            if self.encoders is None:
                logger.error("Encoder not found in the loaded encoders.")
                raise ValueError("Encoder not found in the loaded encoders.")
            logger.info("Encoders loaded successfully")
        except Exception as e:
            logger.error(f"Error loading encoders: {str(e)}")
            raise
        
        # Ensure DataEncoder is fitted before using
        if self.encoders is not None and not self.encoders.fitted:
            logger.info("DataEncoder is not fitted. Fitting now with catalog_data...")
            self.encoders.fit(self.catalog_data)
        
        # Get dimensions through the encoder's get_dims method
        try:
            dims = self.encoders.get_dims()
            self.embedding_dims = {
                'music_dims': dims['music_dims'],
                'artist_dims': dims['artist_dims'],
                'genre_dims': dims['genre_dims'],
                'num_numerical': dims['num_numerical']
            }
            logger.info("Model dimensions loaded successfully")
        except Exception as e:
            logger.error(f"Error getting dimensions: {str(e)}")
            raise
            
        # Safety check for catalog data - update to use encoder's vocabulary
        try:
            max_music_id = self.catalog_data['music'].nunique()
            if max_music_id >= self.embedding_dims['music_dims']:
                logger.warning("Catalog contains more music items than model capacity")
                valid_music = set(self.encoders.music_vectorizer.vocabulary_.keys())
                self.catalog_data = self.catalog_data[
                    self.catalog_data['music'].isin(valid_music)
                ]
                logger.info(f"Filtered catalog size: {len(self.catalog_data)}")
        except Exception as e:
            logger.error(f"Error during catalog validation: {str(e)}")
            raise
        
        self.model = self._initialize_model(self.embedding_dims)
        
    def _initialize_model(self, embedding_dims):
        """Initialize and load the model from checkpoint."""
        model = HybridMusicRecommender(
            music_dims=embedding_dims['music_dims'],
            artist_dims=embedding_dims['artist_dims'],
            genre_dims=embedding_dims['genre_dims'],
            num_numerical=embedding_dims['num_numerical'],
            embedding_dim=self.config.get('embedding_dim', 64),
            layers=self.config.get('hidden_layers', [256, 128, 64]),
            dropout=self.config.get('dropout', 0.2)
        )
        
        # Load state dict from checkpoint
        state_dict = self.checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        
        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def generate_recommendations(self, user_info: dict, n_recommendations: int = 10) -> pd.DataFrame:
        """Generate recommendations with complete feature handling."""
        try:
            # Create a temporary DataFrame with all songs for the user
            user_candidates = self.catalog_data.copy()
            
            # Add user information to all candidates
            user_candidates['age'] = user_info['age']
            user_candidates['gender'] = user_info.get('gender', 'U')
            
            # Handle explicit flag properly
            if 'explicit' in user_candidates.columns:
                user_candidates['explicit'] = user_candidates['explicit'].astype(str)
            else:
                user_candidates['explicit'] = 'False'
            
            # Ensure all required columns are present
            required_cols = self.encoders.numerical_features + ['music', 'artist_name', 'main_genre']
            missing_cols = [col for col in required_cols if col not in user_candidates.columns]
            if missing_cols:
                logger.warning(f"Missing columns in catalog: {missing_cols}")
                for col in missing_cols:
                    user_candidates[col] = 0  # Add default values
            
            # Create dataset with complete features
            test_dataset = MusicRecommenderDataset(
                user_candidates,
                mode='test',
                encoders=self.encoders
            )
            
            # ...rest of the recommendation generation code...
            
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
        
        # Generate explanations for the top recommendations
        try:
            batch = next(iter(test_loader))  # Get the first batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            explanations = self.model.explain_prediction(batch)
        except Exception as e:
            logger.warning(f"Could not generate explanations: {str(e)}")
            explanations = {}
            
        recommendations['explanation'] = recommendations.apply(
            lambda x: self._generate_explanation_text(x, explanations), axis=1
        )
        
        # Convert back to genre names for output
        try:
            recommendations['genre'] = self.encoders.genre_encoder.inverse_transform(
                recommendations['genre'].astype(int)
            )
        except ValueError as ve:
            logger.error(f"Inverse transform failed: {ve}")
            # Handle invalid genres gracefully
            recommendations['genre'] = recommendations['genre'].apply(
                lambda x: self.encoders.genre_encoder.inverse_transform([x])[0] if str(x).isdigit() else 'Unknown'
            )
        
        return recommendations

    def _generate_explanation_text(self, row, explanations):
        """Generate human-readable explanation for recommendation."""
        feature_importance = explanations
        
        explanation = []
        if feature_importance.get('genre', 0) > 0.2:
            explanation.append(f"matches your preferred genre ({row['genre']})")
        if feature_importance.get('artist', 0) > 0.2:
            explanation.append("is similar to artists you like")
        if feature_importance.get('music', 0) > 0.2:
            explanation.append("features music you enjoy")
        if feature_importance.get('numerical', 0) > 0.2:
            explanation.append("aligns with your listening habits")
        if feature_importance.get('binary', 0) > 0.2:
            explanation.append("matches your explicit and gender preferences")
        
        if not explanation:
            explanation.append("matches your overall preferences")
            
        return "Recommended because it " + " and ".join(explanation)

class HybridMusicRecommender(nn.Module):
    def __init__(self, music_dims: int, artist_dims: int, genre_dims: int,
                 num_numerical: int, embedding_dim=64, layers=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        # Linear layers instead of Embeddings to match checkpoint
        self.music_layer = nn.Linear(music_dims, embedding_dim)
        self.artist_layer = nn.Linear(artist_dims, embedding_dim)
        self.genre_layer = nn.Linear(genre_dims, embedding_dim)
        self.genre_embedding = nn.Embedding(genre_dims, embedding_dim)  # Keep this for categorical genre encoding
        
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
        
        # Adjust total features to 5 embeddings * embedding_dim
        total_features = embedding_dim * 5  # music + artist + genre + numerical + binary
        
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
        # Process features using Linear layers
        music_emb = self.music_layer(batch['music_features'].float())
        artist_emb = self.artist_layer(batch['artist_features'].float())
        genre_emb = self.genre_embedding(batch['genre_features'])  # Use embedding for categorical
        
        # Process numerical features
        numerical = self.numerical_layer(batch['numerical_features'])
        
        # Process binary features
        binary = torch.stack([batch['explicit'], batch['gender']], dim=1).float()
        binary = self.binary_layer(binary)
        
        # Concatenate all features
        x = torch.cat([
            music_emb, artist_emb, genre_emb, numerical, binary
        ], dim=1)
        
        # Apply MLP layers with residual connections
        for layer in self.fc_layers:
            identity = x
            x = layer['main'](x)
            if layer['residual'] is not None:
                x = x + layer['residual'](identity)
        
        # Final prediction
        return self.final_layer(x)
    
    def explain_prediction(self, batch):
        """
        Generate feature attributions using Integrated Gradients.
        """
        return super().explain_prediction(batch)  # Utilize the method from train_model.py

def main():
    # Example usage
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = 'checkpoints/best_model.pth'
    catalog_data = pd.read_csv(os.path.join(BASE_DIR, 'data','processed_data', 'test_data.csv'))
    encoders_path = os.path.join(BASE_DIR, 'data', 'processed_data', 'encoder.pt')
    
    # Initialize recommendation generator
    recommender = RecommendationGenerator(model_path, catalog_data, encoders_path)
    
    # Example user with standardized genres
    user_info = {
        'age': 32,
        'gender': 'M',
        'favourite_genres': ['hip hop', 'classical', 'R&B'],  # Will be standardized
        'favourite_artists': ['Gunna', 'Drake'],
        'favourite_music': ['Top Floor', 'Started From the Bottom']
    }
    
    # Print available genres before generating recommendations
    logger.info(f"Available genres: {sorted(recommender.encoders.known_genres)}")
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(user_info, n_recommendations=10)
    
    print("\nTop 10 Recommended Songs:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
