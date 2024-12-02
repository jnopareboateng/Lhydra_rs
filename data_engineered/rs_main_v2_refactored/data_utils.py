import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from encoder_utils import DataEncoder
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_and_save_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets while maintaining consistent encoding.
    """
    # Read data
    df = pd.read_csv(data_path)
    logger.info(f"Total records: {len(df)}")
    
    # Initialize and fit encoders on full dataset
    encoder = DataEncoder()
    encoder.fit(df)
    
    # Split data
    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    
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
    data_path = '/home/josh/Lhydra_rs/data_engineered/rs_main_v2_refactored/data/engineered_data.csv'
    train_path, test_path, encoder_path = split_and_save_data(data_path)
