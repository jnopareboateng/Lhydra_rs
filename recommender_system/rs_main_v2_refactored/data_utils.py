import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from encoder_utils import DataEncoder
import torch
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def augment_numerical_features(row: pd.Series, noise_level: float = 0.1) -> pd.Series:
    """Add small gaussian noise to numerical features."""
    numerical_cols = [
        'duration', 'acousticness', 'key', 'mode', 'speechiness',
        'instrumentalness', 'liveness', 'tempo', 'time_signature',
        'energy_loudness', 'dance_valence'
    ]
    
    augmented = row.copy()
    for col in numerical_cols:
        if col in row:
            noise = np.random.normal(0, noise_level * np.abs(row[col]))
            augmented[col] = row[col] + noise
    
    return augmented

def balance_classes(df: pd.DataFrame, target_col: str = 'main_genre', 
                   min_samples: int = None, max_samples: int = None) -> pd.DataFrame:
    """Balance dataset by genre through augmentation and/or subsampling."""
    class_counts = df[target_col].value_counts()
    
    if not min_samples:
        min_samples = class_counts.min()
    if not max_samples:
        max_samples = class_counts.max()
    
    balanced_dfs = []
    
    for genre in class_counts.index:
        genre_df = df[df[target_col] == genre]
        n_samples = len(genre_df)
        
        if n_samples < min_samples:
            # Augment underrepresented classes
            n_augment = min_samples - n_samples
            augmented_samples = []
            
            for _ in range(n_augment):
                base_row = genre_df.sample(n=1).iloc[0]
                augmented_row = augment_numerical_features(base_row)
                augmented_samples.append(augmented_row)
            
            genre_df = pd.concat([genre_df, pd.DataFrame(augmented_samples)], ignore_index=True)
        
        elif n_samples > max_samples:
            # Subsample overrepresented classes
            genre_df = genre_df.sample(n=max_samples, random_state=42)
        
        balanced_dfs.append(genre_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def generate_synthetic_samples(df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic samples using feature distributions."""
    synthetic_samples = []
    
    # Get unique values for categorical features
    unique_genres = df['main_genre'].unique()
    unique_artists = df['artist_name'].unique()
    
    # Fit scalers for numerical features
    numerical_cols = [
        'duration', 'acousticness', 'key', 'mode', 'speechiness',
        'instrumentalness', 'liveness', 'tempo', 'time_signature',
        'energy_loudness', 'dance_valence'
    ]
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[numerical_cols])
    
    # Generate synthetic samples
    for _ in range(n_samples):
        sample = {}
        
        # Sample categorical features
        sample['main_genre'] = np.random.choice(unique_genres)
        sample['artist_name'] = np.random.choice(unique_artists)
        sample['music'] = f"synthetic_track_{_}"
        
        # Generate numerical features
        synthetic_numericals = np.random.normal(
            loc=scaled_features.mean(axis=0),
            scale=scaled_features.std(axis=0)
        )
        synthetic_numericals = scaler.inverse_transform(synthetic_numericals.reshape(1, -1))[0]
        
        for col, value in zip(numerical_cols, synthetic_numericals):
            sample[col] = value
        
        synthetic_samples.append(sample)
    
    synthetic_df = pd.DataFrame(synthetic_samples)
    return synthetic_df


def generate_negative_samples(df: pd.DataFrame, num_negatives: int = 4,
                            balance_negatives: bool = True) -> pd.DataFrame:
    """Generate negative samples with optional balancing."""
    unique_music = set(df['music'].unique())
    unique_artists = set(df['artist_name'].unique())
    unique_genres = set(df['main_genre'].unique())
    
    negative_samples = []
    for _, user_data in tqdm(df.groupby(['age', 'gender'])):  # Group by age and gender
        # Get this group's preferences
        user_music = set(user_data['music'])
        
        # Sample from items not in user's history
        available_music = list(unique_music - user_music)
        
        if available_music:  # Only proceed if we have available music
            # Sample negative items
            num_samples = min(len(available_music), num_negatives)
            sampled_music = np.random.choice(available_music, num_samples, replace=False)
            
            # Create negative samples
            base_data = user_data.iloc[0].to_dict()
            for music in sampled_music:
                negative = base_data.copy()
                negative['music'] = music
                # Use original artist and genre mapping from the music catalog
                music_info = df[df['music'] == music].iloc[0]
                negative['artist_name'] = music_info['artist_name']
                negative['main_genre'] = music_info['main_genre']
                negative['playcount'] = 0  # Mark as negative sample
                negative_samples.append(negative)
    
    if not negative_samples:
        logger.warning("No negative samples generated!")
        return df
        
    negative_df = pd.DataFrame(negative_samples)
    
    if balance_negatives:
        # Balance positive and negative samples
        n_positives = len(df)
        neg_sample_weight = n_positives / len(negative_df)
        negative_df['sample_weight'] = neg_sample_weight
        df['sample_weight'] = 1.0
    
    return pd.concat([df, negative_df], ignore_index=True)

def split_and_save_data(data_path: str, test_size: float = 0.2, random_state: int = 42,
                       balance_data: bool = True, augment_data: bool = True):
    """Split data into train and test sets with optional balancing and augmentation."""
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    
    # Ensure required columns exist
    required_cols = ['music', 'artist_name', 'main_genre', 'age', 'gender', 'playcount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique music items: {df['music'].nunique()}")
    logger.info(f"Unique artists: {df['artist_name'].nunique()}")
    logger.info(f"Unique genres: {df['main_genre'].nunique()}")
    
    # Generate negative samples
    df = generate_negative_samples(df)
    logger.info(f"Total records after negative sampling: {len(df)}")
    
    if balance_data:
        df = balance_classes(df)
        logger.info(f"\nBalanced dataset size: {len(df)}")
        logger.info(f"Balanced genre distribution:\n{df['main_genre'].value_counts()}")
    
    if augment_data and len(df) < 10000:  # Only augment if dataset is small
        synthetic_df = generate_synthetic_samples(df)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        logger.info(f"\nAugmented dataset size: {len(df)}")
    
    # Split by age groups instead of users
    age_groups = pd.qcut(df['age'], q=5)
    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=age_groups  # Stratify by age groups
    )
    
    # Normalize genres before fitting encoder
    encoder = DataEncoder(max_features=5000)
    df['main_genre'] = df['main_genre'].apply(encoder.normalize_genre)
    
    # Log unique genres after normalization
    logger.info(f"Unique genres after normalization: {sorted(df['main_genre'].unique())}")
    
    # Initialize encoder with increased capacity
    encoder = DataEncoder(max_features=5000)
    
    # Fit on full dataset to capture all items
    encoder.fit(pd.concat([train_data, test_data]))
    
    # Verify vocabulary coverage
    logger.info(f"Music vocabulary size: {len(encoder.music_vectorizer.vocabulary_)}")
    logger.info(f"Artist vocabulary size: {len(encoder.artist_vectorizer.vocabulary_)}")
    logger.info(f"Genre vocabulary size: {len(encoder.genre_encoder.classes_)}")
    
    # Save with more descriptive names and ensure directories exist
    base_dir = os.path.dirname(data_path)
    data_dir = os.path.join(base_dir, 'processed_data')
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    encoder_path = os.path.join(data_dir, 'encoder.pt')
    
    # Save data and encoders
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    torch.save({'encoder': encoder}, encoder_path)
    
    # Log feature dimensions
    dims = encoder.get_dims()
    logger.info("\nFeature dimensions:")
    for key, value in dims.items():
        logger.info(f"{key}: {value}")
    
    return train_path, test_path, encoder_path

def validate_feature_compatibility(data: pd.DataFrame, encoders: DataEncoder) -> bool:
    """Validate that data has all required features for model."""
    required_cols = (
        encoders.numerical_features +
        ['music', 'artist_name', 'main_genre', 'explicit', 'gender']
    )
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
        
    try:
        # Test transform to catch any incompatibilities
        features = encoders.transform(data.head(1))
        required_features = [
            'music_features', 'artist_features', 'genre_features',
            'numerical_features', 'explicit', 'gender'
        ]
        
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            logger.error(f"Missing features after transform: {missing_features}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating features: {str(e)}")
        return False

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data to ensure all required features exist."""
    df = df.copy()
    
    # Handle explicit content
    if 'explicit' not in df.columns:
        df['explicit'] = False
    
    # Handle gender
    if 'gender' not in df.columns:
        df['gender'] = 'U'
    
    # Handle numerical features
    numerical_cols = [
        'duration', 'acousticness', 'key', 'mode', 'speechiness',
        'instrumentalness', 'liveness', 'tempo', 'time_signature',
        'energy_loudness', 'dance_valence'
    ]
    
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df

if __name__ == "__main__":
    data_path = '../../data/o2_data.csv'
    train_path, test_path, encoder_path = split_and_save_data(data_path)

import unittest

class TestDataEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = DataEncoder()
        sample_data = pd.DataFrame({
            'music': ['Shape of You', 'Hello'],
            'artist_name': ['Ed Sheeran', 'Adele'],
            'main_genre': ['pop', 'hip-hop'],
            'age': [25, 30],
            'duration': [210, 200],
            'acousticness': [0.5, 0.3],
            'key': [5, 7],
            'mode': [1, 0],
            'speechiness': [0.04, 0.05],
            'instrumentalness': [0.0, 0.0],
            'liveness': [0.1, 0.2],
            'tempo': [120.0, 130.0],
            'time_signature': [4, 4],
            'energy_loudness': [0.8, 0.7],
            'dance_valence': [0.9, 0.85]
        })
        self.encoder.fit(sample_data)

    def test_normalize_genre(self):
        self.assertEqual(self.encoder.normalize_genre('pop'), 'Pop')
        self.assertEqual(self.encoder.normalize_genre('hip-hop'), 'Hip-Hop')
        self.assertEqual(self.encoder.normalize_genre('unknown_genre'), 'Other')

    def test_transform_genres(self):
        genres = ['pop', 'hip-hop', 'electronic']
        transformed = self.encoder.transform_genres(genres)
        # Assuming 'electronic' was mapped correctly or defaulted
        self.assertTrue(len(transformed) == 3)

    def test_transform(self):
        test_df = pd.DataFrame({
            'music': ['Shape of You', 'Unknown Song'],
            'artist_name': ['Ed Sheeran', 'Unknown Artist'],
            'main_genre': ['pop', 'unknown_genre'],
            'age': [25, 30],
            'duration': [210, 200],
            'acousticness': [0.5, 0.3],
            'key': [5, 7],
            'mode': [1, 0],
            'speechiness': [0.04, 0.05],
            'instrumentalness': [0.0, 0.0],
            'liveness': [0.1, 0.2],
            'tempo': [120.0, 130.0],
            'time_signature': [4, 4],
            'energy_loudness': [0.8, 0.7],
            'dance_valence': [0.9, 0.85]
        })
        transformed = self.encoder.transform(test_df)
        self.assertIn('music_features', transformed)
        self.assertIn('artist_features', transformed)
        self.assertIn('genre_features', transformed)
        self.assertIn('numerical_features', transformed)
