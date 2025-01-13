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
        self.binary_features = ['explicit', 'gender']
        
        # Initialize binary encoders
        self.explicit_map = {'True': 1, 'False': 0, True: 1, False: 0}
        self.gender_map = {'M': 1, 'F': 0}
        
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
        
        # Add 'Other' to genre mapping
        self.genre_mapping.update({
            'other': 'Other',
            'unknown': 'Other',
            '': 'Other'
        })
        
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
        
        # Ensure 'Other' category is present
        if 'Other' not in df['main_genre'].values:
            df = pd.concat([df, pd.DataFrame([{'main_genre': 'Other'}])], ignore_index=True)
        
        # Log unique genres before normalization
        logger.info(f"Unique genres before normalization: {df['main_genre'].unique()}")
        
        df['main_genre'] = df['main_genre'].apply(self.normalize_genre)
        
        # Log unique genres after normalization
        logger.info(f"Unique genres after normalization: {df['main_genre'].unique()}")
        
        self.known_genres = set(df['main_genre'].str.lower().unique())
        self.genres_classes_ = sorted(list(self.known_genres))  # Ensure consistent ordering
        
        # Convert to string to handle any numerical IDs
        self.music_vectorizer.fit(df['music'].astype(str))
        self.artist_vectorizer.fit(df['artist_name'].astype(str))
        self.genre_encoder.fit(df['main_genre'])
        
        # Scale numerical features with feature names
        if len(self.numerical_features) > 0:
            numerical_df = df[self.numerical_features].copy()
            self.scaler.fit(numerical_df)
            # Store feature names used during fitting
            self.numerical_feature_names_ = numerical_df.columns.tolist()
        
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
        """Transform data with complete feature set."""
        if not self.fitted:
            raise ValueError("DataEncoder must be fitted before calling transform")
            
        try:
            # Handle missing columns with defaults
            df = df.copy()
            
            # Handle explicit column properly
            if 'explicit' not in df.columns:
                df['explicit'] = False
            else:
                # Convert to string first to handle various input types
                df['explicit'] = df['explicit'].astype(str).map(
                    lambda x: self.explicit_map.get(x.lower(), 0)
                )
            
            # Handle gender with proper defaults
            if 'gender' not in df.columns:
                df['gender'] = 'U'
            df['gender'] = df['gender'].fillna('U')
            
            # Transform categorical features
            music_features = self.music_vectorizer.transform(
                df['music'].fillna('').astype(str)
            )
            artist_features = self.artist_vectorizer.transform(
                df['artist_name'].fillna('').astype(str)
            )
            genre_features = self.transform_genres(
                df['main_genre'].fillna('Other').tolist()
            )
            
            # Transform numerical features using DataFrame to preserve feature names
            numerical_df = df[self.numerical_features].fillna(
                df[self.numerical_features].mean()
            )
            numerical_features = self.scaler.transform(numerical_df)
            
            # Transform binary features with proper type conversion
            explicit = np.array(df['explicit'].values, dtype=np.float32)
            gender = np.array([
                self.gender_map.get(str(x).upper(), 0) 
                for x in df['gender']
            ], dtype=np.float32)
            
            return {
                'music_features': music_features,
                'artist_features': artist_features,
                'genre_features': genre_features,
                'numerical_features': numerical_features,
                'explicit': explicit,
                'gender': gender
            }
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}\nDataframe columns: {df.columns}")
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

