from sklearn.preprocessing import LabelEncoder, RobustScaler
import pandas as pd
from typing import Dict, Any
import numpy as np

class DataEncoder:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.music_encoder = LabelEncoder()
        self.artist_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        
        self.numerical_features = [
            'age', 'duration', 'acousticness', 'key', 'mode', 'speechiness',
            'instrumentalness', 'liveness', 'tempo', 'time_signature',
            'energy_loudness', 'dance_valence'  # Removed 'playcount'
        ]
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit all encoders on the full dataset."""
        self.user_encoder.fit(df['user_id'].values)
        self.music_encoder.fit(df['music_id'].values)
        self.artist_encoder.fit(df['artist_id'].values)
        self.genre_encoder.fit(df['main_genre'].values)
        self.scaler.fit(df[self.numerical_features].values)
    
    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform data using fitted encoders."""
        return {
            'users': self.user_encoder.transform(df['user_id'].values),
            'music': self.music_encoder.transform(df['music_id'].values),
            'artists': self.artist_encoder.transform(df['artist_id'].values),
            'genres': self.genre_encoder.transform(df['main_genre'].values),
            'numerical_features': self.scaler.transform(df[self.numerical_features].values)
        }
    
    def get_dims(self) -> Dict[str, int]:
        """Get dimensions for model initialization."""
        return {
            'num_users': len(self.user_encoder.classes_),
            'num_music': len(self.music_encoder.classes_),
            'num_artists': len(self.artist_encoder.classes_),
            'num_genres': len(self.genre_encoder.classes_),
            'num_numerical': len(self.numerical_features)
        }
    
    def get_encoders(self) -> Dict[str, Any]:
        """Get all encoders for saving."""
        return {
            'user_encoder': self.user_encoder,
            'music_encoder': self.music_encoder,
            'artist_encoder': self.artist_encoder,
            'genre_encoder': self.genre_encoder,
            'scaler': self.scaler
        }
