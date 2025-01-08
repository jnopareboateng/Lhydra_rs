from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, LabelEncoder
import pandas as pd
from typing import Dict, Any
import numpy as np
import logging
from collections import defaultdict
from genre_parser import GenreParser
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataEncoder:
    def __init__(self, max_features=5000):
        # Correct the base_dir to the current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        genres_file = os.path.join(base_dir, 'all_genres.txt')
        
        # Parse genres from file
        parser = GenreParser(genres_file)
        self.MAIN_GENRES = parser.get_main_genres_dict()
        logger.info(f"MAIN_GENRES loaded with categories: {list(self.MAIN_GENRES.keys())}")
        
        # Initialize all vectorizers and encoders
        self.music_vectorizer = TfidfVectorizer(max_features=max_features)
        self.artist_vectorizer = TfidfVectorizer(max_features=max_features//2)
        self.genre_encoder = LabelEncoder()  # Initialize genre encoder
        
        self.scaler = RobustScaler()
        
        # Initialize class attributes
        self._is_fitted = False
        self.genres_classes_ = None
        
        self.numerical_features = [
            'age', 'duration', 'acousticness', 'key', 'mode', 'speechiness',
            'instrumentalness', 'liveness', 'tempo', 'time_signature',
            'energy_loudness', 'dance_valence'
        ]
        
        # Add genre mapping dictionary for normalization
        self.genre_mapping = {
            'pop': 'Pop',
            'rock': 'Rock',
            'hip hop': 'Hip-Hop',
            'hip-hop': 'Hip-Hop',
            'hip-hop/rap': 'Hip-Hop/Rap',
            'r&b': 'R&B',
            'rap': 'Rap',
            # Add more mappings as needed
        }
        
        # Add genre clustering fields
        self.genre_map = self._create_genre_map()
        logger.info(f"Genre map initialized with {len(self.genre_map)} entries.")
        logger.debug(f"Genre map contents: {self.genre_map}")
        
        self.known_genres = set()
        
    def _create_genre_map(self):
        """Create mapping of subgenres to main genres"""
        genre_map = {}
        for main_genre, subgenres in self.MAIN_GENRES.items():
            for subgenre in subgenres:
                # Store subgenres in lowercase to match classification
                genre_map[subgenre.lower()] = main_genre.title()
            # Ensure the main genre itself is included for exact matching
            genre_map[main_genre.lower()] = main_genre.title()
        logger.info(f"Created genre_map with {len(genre_map)} mappings.")
        logger.debug(f"Genre_map details: {genre_map}")
        return genre_map
        
    def classify_genre(self, genre: str) -> str:
        """Map a genre to its main category"""
        genre = str(genre).lower().strip()
        logger.debug(f"Classifying genre: '{genre}'")
        
        # Exact match first
        if genre in self.genre_map:
            logger.debug(f"Exact match found for genre '{genre}': '{self.genre_map[genre]}'")
            return self.genre_map[genre]
        
        # Partial match
        for key_term, main_genre in self.genre_map.items():
            if key_term in genre:
                logger.debug(f"Partial match found for genre '{genre}': '{main_genre}'")
                return main_genre
        
        logger.debug(f"No match found for genre '{genre}', defaulting to 'other'")
        return "other"
    
    @property
    def fitted(self):
        return (hasattr(self, '_is_fitted')
                and self._is_fitted
                and hasattr(self.genre_encoder, 'classes_')
                and hasattr(self.music_vectorizer, 'vocabulary_')
                and hasattr(self.artist_vectorizer, 'vocabulary_'))
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit vectorizers and encoders on the full dataset."""
        df = df.copy()
        # Log unique genres before normalization
        logger.info(f"Unique genres before normalization: {df['main_genre'].unique()}")
        
        df['main_genre'] = df['main_genre'].apply(self.normalize_genre)
        
        # Log unique genres after normalization
        logger.info(f"Unique genres after normalization: {df['main_genre'].unique()}")
        
        self.known_genres = set(df['main_genre'].str.lower().unique())
        logger.info(f"Known genres after fitting: {sorted(self.known_genres)}")
        logger.debug(f"Genres classes: {self.genres_classes_}")
        
        # Convert to string to handle any numerical IDs
        self.music_vectorizer.fit(df['music'].astype(str))
        self.artist_vectorizer.fit(df['artist_name'].astype(str))
        self.genre_encoder.fit(df['main_genre'])
        self.genres_classes_ = self.genre_encoder.classes_
        
        if len(self.numerical_features) > 0:
            self.scaler.fit(df[self.numerical_features].values)
        
        self._is_fitted = True
    
    def normalize_genre(self, genre: str) -> str:
        """Normalize genre names to standard format"""
        if not genre:
            logger.debug("Empty genre found, defaulting to 'other'")
            return "other"
            
        normalized = self.classify_genre(genre)
        logger.debug(f"Normalizing genre '{genre}' to '{normalized}'")
        return normalized.title()  # Return capitalized version
    
    def transform_genres(self, genres: list) -> np.ndarray:
        """Transform list of genres with standardization"""
        # Normalize and standardize genres
        normalized_genres = [self.normalize_genre(g) for g in genres]
        
        # Filter to known genres and handle unknowns
        known_genres = [g for g in normalized_genres if g.lower() in self.known_genres]
        
        if not known_genres:
            # Use most common genre as fallback
            fallback = self.genres_classes_[0] if self.genres_classes_ is not None else "Other"
            logger.warning(f"No known genres in {genres}, using {fallback}")
            known_genres = [fallback]
            
        try:
            return self.genre_encoder.transform(known_genres)
        except Exception as e:
            logger.error(f"Error transforming genres {known_genres}: {str(e)}")
            return np.array([0])  # Fallback to first genre index
    
    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform data using fitted encoders with safety checks."""
        if not self.fitted:
            raise ValueError("DataEncoder must be fitted before calling transform")
        
        try:
            # Handle missing columns by filling with defaults
            for col in ['music', 'artist_name', 'main_genre']:
                if col not in df.columns:
                    df[col] = ''  # Empty string as default
                    
            # Convert to string and handle NaN values
            music_str = df['music'].fillna('').astype(str)
            artist_str = df['artist_name'].fillna('').astype(str)
            genre_str = df['main_genre'].fillna(self.genres_classes_[0]).astype(str)
            
            return {
                'music_features': self.music_vectorizer.transform(music_str),
                'artist_features': self.artist_vectorizer.transform(artist_str),
                'genre_features': self.transform_genres(genre_str),
                'numerical_features': self.scaler.transform(df[self.numerical_features].fillna(0).values) 
                    if len(self.numerical_features) > 0 else np.array([])
            }
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
    
    def get_dims(self) -> Dict[str, int]:
        """Get dimensions for model initialization."""
        if not self.fitted:
            raise ValueError("DataEncoder must be fitted before getting dimensions")
            
        return {
            'music_dims': self.music_vectorizer.max_features,
            'artist_dims': self.artist_vectorizer.max_features,
            'genre_dims': len(self.genres_classes_),
            'num_numerical': len(self.numerical_features)
        }
    
    def get_encoders(self) -> Dict[str, Any]:
        """Get encoder instance for saving."""
        return {
            'encoder': self  # Return self as the encoder
        }
    
    def inspect_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with original and normalized genres for inspection.
        This helps verify how subgenres are mapped to main genres.
        """
        df = df.copy()
        df['normalized_genre'] = df['main_genre'].apply(self.normalize_genre)
        logger.info("Inspection of genre normalization:")
        logger.info(df[['main_genre', 'normalized_genre']].drop_duplicates())
        logger.debug(df[['main_genre', 'normalized_genre']].drop_duplicates())
        return df[['main_genre', 'normalized_genre']].drop_duplicates()

