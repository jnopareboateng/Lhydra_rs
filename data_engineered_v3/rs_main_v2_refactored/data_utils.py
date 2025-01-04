import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from encoder_utils import DataEncoder
import torch
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_negative_samples(df: pd.DataFrame, num_negatives: int = 4) -> pd.DataFrame:
    """
    Generate negative samples for each user by randomly sampling items they haven't interacted with.
    
    Args:
        df: DataFrame containing user-item interactions
        num_negatives: Number of negative samples per positive interaction
    
    Returns:
        DataFrame with both positive and negative samples
    """
    # Create a set of all items
    all_items = set(df['music_id'].unique())
    
    negative_samples = []
    for user_id in tqdm(df['user_id'].unique(), desc="Generating negative samples"):
        # Get items the user has interacted with
        user_items = set(df[df['user_id'] == user_id]['music_id'])
        
        # Get items the user hasn't interacted with
        negative_items = list(all_items - user_items)
        
        if len(negative_items) > 0:
            # Sample negative items
            num_samples = min(len(negative_items), num_negatives)
            sampled_negatives = np.random.choice(negative_items, size=num_samples, replace=False)
            
            # Create negative samples
            user_data = df[df['user_id'] == user_id].iloc[0].to_dict()
            for item_id in sampled_negatives:
                negative = user_data.copy()
                negative['music_id'] = item_id
                negative['playcount'] = 0  # Mark as negative sample
                negative_samples.append(negative)
    
    # Convert negative samples to DataFrame
    negative_df = pd.DataFrame(negative_samples)
    
    # Combine positive and negative samples
    combined_df = pd.concat([df, negative_df], ignore_index=True)
    
    return combined_df

def split_and_save_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets while maintaining consistent encoding.
    """
    # Read data
    df = pd.read_csv(data_path)
    logger.info(f"Total records: {len(df)}")
    
    # Generate negative samples
    df = generate_negative_samples(df)
    logger.info(f"Total records after negative sampling: {len(df)}")
    
    # Initialize and fit encoders on full dataset
    encoder = DataEncoder()
    encoder.fit(df)
    
    # Split by user to avoid data leakage
    users = df['user_id'].unique()
    train_users, test_users = train_test_split(
        users,
        test_size=test_size,
        random_state=random_state
    )
    
    train_data = df[df['user_id'].isin(train_users)]
    test_data = df[df['user_id'].isin(test_users)]
    
    # Save splits
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    encoder_path = os.path.join(data_dir, 'data_encoders.pt')
    
    # Save data splits
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Save encoders
    torch.save(encoder.get_encoders(), encoder_path)
    
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    logger.info(f"\nFiles saved to:")
    logger.info(f"Training data: {train_path}")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Encoders: {encoder_path}")
    
    # Log some statistics about the encodings
    dims = encoder.get_dims()
    logger.info("\nEncoding dimensions:")
    for key, value in dims.items():
        logger.info(f"{key}: {value}")
    
    return train_path, test_path, encoder_path

if __name__ == "__main__":
    data_path = '../../data/o2_data.csv'
    train_path, test_path, encoder_path = split_and_save_data(data_path)
