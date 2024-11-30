import pandas as pd
from tqdm import tqdm
import torch
from aio import (
    FeatureProcessor,
    MultiTaskMusicRecommender,
    SongEmbeddingRetriever,
    MusicRecommenderSystem,
    DataChunkLoader,
    train_model,
    evaluate_model
)
import numpy as np
import os
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
import logging

# Define features
NUMERICAL_COLS = [
    "age",
    "duration_ms",
    "loudness",
    "instrumentalness",
    "liveness",
    "tempo",
    "energy",
    "danceability",
    "valence",
]
CATEGORICAL_COLS = ["gender", "genre", "artist", "track_id"]

# Initialize components
feature_processor = FeatureProcessor(NUMERICAL_COLS, CATEGORICAL_COLS)
chunk_loader = DataChunkLoader(
    "../data/msd_full_data_processed.csv", chunk_size=100_000
)

# Fit feature processor on chunks
print("Fitting feature processor on entire dataset...")
for chunk in tqdm(chunk_loader.load_chunks()):
    feature_processor.fit(chunk)


# Split data into train and validation sets
def split_and_save_data(
    input_file: str,
    train_file: str = "train_data.csv",
    val_file: str = "val_data.csv",
    val_ratio: float = 0.2,
    chunk_size: int = 100_000,
    random_seed: int = 42,
    force_replace: bool = False,
):
    # Read the entire dataset (if it fits into memory)
    data = pd.read_csv(input_file)
    # Shuffle the data
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # Split the data
    val_size = int(len(data) * val_ratio)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    # Save to CSV
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)


# Split data into train and validation sets
split_and_save_data(
    input_file="../data/msd_full_data_processed.csv",
    train_file="train_data.csv",
    val_file="val_data.csv",
    val_ratio=0.2,
    chunk_size=100_000,
)

# Initialize model
model = MultiTaskMusicRecommender(
    feature_processor=feature_processor,
    n_genres=len(feature_processor.encoders["genre"].classes_),
)

# Initialize song retriever
song_retriever = SongEmbeddingRetriever(n_trees=10)

# Create recommendation system
recommender = MusicRecommenderSystem(
    model=model, feature_processor=feature_processor, song_retriever=song_retriever
)

# Add device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device and set device attribute
model = model.to(device)
model.device = device

# Add target columns to the feature processing
TARGET_COLS = ["playcount", "genre", "ranking"]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.length = len(next(iter(features.values())))
        # Verify lengths
        for v in features.values():
            assert len(v) == self.length
        for v in targets.values():
            assert len(v) == self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.features.items()}
        item.update({k: v[idx] for k, v in self.targets.items()})
        return item


def create_dataloader(df: pd.DataFrame, feature_processor: FeatureProcessor, batch_size: int = 64):
    if 'ranking' not in df.columns:
        print("Warning: 'ranking' column is missing. Assigning default values.")
        df['ranking'] = 0.0  # Assign a default value

    processed_features = feature_processor.transform(df)
    print(f"Processed feature keys: {list(processed_features.keys())}")
    
    targets = {
        'playcount': df['playcount'].values.astype(np.float32).reshape(-1, 1),  # Add reshape
        'genre': processed_features['genre_encoded'].numpy().astype(np.int64),
        'ranking': df['ranking'].values.astype(np.float32).reshape(-1, 1)  # Reshape to [N, 1]
    }

    dataset = CustomDataset(processed_features, targets)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,        # Shuffle the data if needed
        # num_workers=4        # Adjust based on your system's capabilities
    )

# Training configuration
num_epochs = 10
best_metrics = None
best_model_path = "best_model.pth"

# Training on chunks
print("Starting training...")
for epoch in range(num_epochs):
    logging.info(f"Starting Epoch {epoch+1}/{num_epochs}")

    # Reset chunk loaders for each epoch
    train_chunks = DataChunkLoader("train_data.csv", chunk_size=100_000)
    val_chunks = DataChunkLoader("val_data.csv", chunk_size=100_000)

    epoch_metrics = defaultdict(list)
    
    # Create progress bar for chunks
    total_chunks = min(train_chunks.total_chunks, val_chunks.total_chunks)
    chunk_iterator = zip(train_chunks.load_chunks(), val_chunks.load_chunks())
    pbar = tqdm(chunk_iterator, total=total_chunks, desc=f"Epoch {epoch+1} Chunks")
    
    for train_chunk, val_chunk in pbar:
        # Process training chunk
        logging.info(f"Processing training chunk of size {len(train_chunk)}")
        train_chunk["genre_encoded"] = feature_processor.encoders["genre"].transform(
            train_chunk["genre"].astype(str)
        )
        train_loader = create_dataloader(train_chunk, feature_processor, batch_size=64)

        # Process validation chunk
        logging.info(f"Processing validation chunk of size {len(val_chunk)}")
        val_chunk["genre_encoded"] = feature_processor.encoders["genre"].transform(
            val_chunk["genre"].astype(str)
        )
        val_loader = create_dataloader(val_chunk, feature_processor, batch_size=64)

        # Train and evaluate
        train_model(model, train_loader, val_loader, device=device)
        chunk_metrics = evaluate_model(model, val_loader)
        
        # Update progress bar with current metrics
        pbar.set_postfix({k: f"{v:.4f}" for k, v in chunk_metrics.items()})
        
        for metric, value in chunk_metrics.items():
            epoch_metrics[metric].append(value)
    
    # Calculate and log average metrics for the epoch
    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
    logging.info(f"Epoch {epoch+1} Validation Metrics:")
    for metric, value in avg_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save best model based on NDCG@10
    if best_metrics is None or avg_metrics['ndcg@10'] > best_metrics['ndcg@10']:
        best_metrics = avg_metrics
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved with NDCG@10: {avg_metrics['ndcg@10']:.4f}")

# Load best model for final use
model.load_state_dict(torch.load(best_model_path))
logging.info(f"Best model loaded with metrics: {best_metrics}")

# Continue with building song embeddings...

# Build song embeddings and the Annoy index
print("Building song embeddings and Annoy index...")
song_embeddings = []
song_ids = []

data_loader = DataChunkLoader("train_data.csv", chunk_size=100_000)
for chunk in tqdm(data_loader.load_chunks(), desc="Processing Chunks for Annoy Index"):
    # Process features for the songs
    song_features = feature_processor.transform(chunk)
    with torch.no_grad():
        outputs = model({k: v.to(device) for k, v in song_features.items()})
        embeddings = outputs["embedding"].cpu().numpy()
        song_embeddings.append(embeddings)
        song_ids.extend(chunk["track_id"].astype(str).tolist())

# Concatenate all embeddings
song_embeddings = np.vstack(song_embeddings)

# Build the Annoy index
song_retriever.build_index(song_embeddings, song_ids)

# Get recommendations for a user
class AudioFeatureExtractor:
    def __init__(self):
        # Read credentials from config file
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Initialize Spotify client
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=config['SPOTIFY']['client_id'],
                client_secret=config['SPOTIFY']['client_secret']
            )
        )

    def get_audio_features(self, track_id: str) -> dict:
        """Extract audio features from Spotify track ID."""
        try:
            # Get audio features from Spotify
            features = self.sp.audio_features(track_id)[0]
            
            # Extract relevant features
            return {
                "duration_ms": features["duration_ms"],
                "loudness": features["loudness"],
                "instrumentalness": features["instrumentalness"],
                "liveness": features["liveness"],
                "tempo": features["tempo"],
                "energy": features["energy"],
                "danceability": features["danceability"],
                "valence": features["valence"]
            }
        except Exception as e:
            print(f"Error fetching audio features: {e}")
            return None

# Usage example:
def create_user_data(track_id: str, user_age: int, user_gender: str):
    # Initialize feature extractor with your Spotify API credentials
    extractor = AudioFeatureExtractor(
        client_id="your_client_id",
        client_secret="your_client_secret"
    )
    
    # Get audio features
    audio_features = extractor.get_audio_features(track_id)
    
    if audio_features is None:
        raise ValueError("Could not fetch audio features for track")
    
    # Create user data DataFrame
    user_data = pd.DataFrame({
        # User demographic features
        "age": [user_age],
        "gender": [user_gender],
        
        # Audio features (automatically fetched)
        **{k: [v] for k, v in audio_features.items()},
        
        # Track information
        "track_id": [track_id]
    })
    
    return user_data

# Example usage:
user_data = create_user_data(
    track_id="spotify:track:27NovPIUIRrOZoCHxABJwK",  # Bank Account by 21 Savage
    user_age=25,
    user_gender="M"
)

recommendations = recommender.get_recommendations(user_data, n_recommendations=10)
print("Recommendations:", recommendations)
