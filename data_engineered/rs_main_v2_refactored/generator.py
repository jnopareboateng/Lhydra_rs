import torch
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor
from model import HybridRecommender
import librosa

# Add missing NUMERICAL_COLS definition
NUMERICAL_COLS = [
    "age", "duration", "acousticness", "key", "mode", "speechiness",
    "instrumentalness", "liveness", "tempo", "time_signature", "explicit",
    "music_age", "energy_loudness", "dance_valence"
]

def load_model_and_preprocessor(model_path, preprocessor_path="models/"):
    """Load the trained model and preprocessor."""
    preprocessor = DataPreprocessor(
        test_size=0.2,
        random_state=42,
        max_artist_features=3995,
        max_genre_features=21,
        max_music_features=5784,
    )
    preprocessor.load_preprocessors(preprocessor_path)

    # Initialize model with correct dimensions
    model = HybridRecommender(
        num_users=len(preprocessor.user_id_encoder.classes_),
        num_genders=len(preprocessor.gender_encoder.classes_),
        num_music_items=len(preprocessor.music_tfidf_vectorizer.vocabulary_),
        num_genres=len(preprocessor.genre_tfidf_vectorizer.vocabulary_),
        num_artist_features=len(preprocessor.artist_tfidf_vectorizer.vocabulary_),
        num_numerical_features=len(NUMERICAL_COLS),
        num_release_years=len(preprocessor.release_year_encoder.categories_[0]),
        embedding_dim=64
    )

    model.load_state_dict(torch.load(model_path))
    return model, preprocessor

def prepare_inference_features(data, preprocessor):
    """Prepare features for inference."""
    # Encode features
    data_encoded = preprocessor.encode_features_transform(data)
    # Apply feature engineering
    features = preprocessor.feature_engineering(data_encoded)
    return features

def get_recommendations(model, user_data, preprocessor, device, top_k=10):
    """Generate recommendations for a user."""
    model.eval()
    features = prepare_inference_features(user_data, preprocessor)

    # Convert features to tensors
    tensor_data = {
        "user_id": torch.tensor(features['user_id_encoded'].values, dtype=torch.long).to(device),
        "gender_ids": torch.tensor(features['gender_encoded'].values, dtype=torch.long).to(device),
        "genre_features": torch.tensor(
            features[[col for col in features.columns if col.startswith('genre_tfidf_')]].values,
            dtype=torch.float32
        ).to(device),
        "artist_features": torch.tensor(
            features[[col for col in features.columns if col.startswith('artist_tfidf_')]].values,
            dtype=torch.float32
        ).to(device),
        "music_features": torch.tensor(
            features[[col for col in features.columns if col.startswith('music_tfidf_')]].values,
            dtype=torch.float32
        ).to(device),
        "numerical_features": torch.tensor(
            features[NUMERICAL_COLS].values,
            dtype=torch.float32
        ).to(device),
        "release_years": torch.tensor(features['release_year_encoded'].values, dtype=torch.long).to(device),
    }

    # Get predictions using correct parameter order
    with torch.no_grad():
        predictions = model(
            tensor_data["user_id"],
            tensor_data["artist_features"],
            tensor_data["gender_ids"],
            tensor_data["music_features"],
            tensor_data["genre_features"],
            tensor_data["numerical_features"],
            tensor_data["release_years"]
        )

    predictions = predictions.cpu().numpy().flatten()
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    recommendations = user_data.iloc[top_indices].copy()
    recommendations["score"] = predictions[top_indices]
    
    return recommendations[["artist_name", "music", "genre", "score"]]

def extract_audio_features(audio_path):
    """Extract audio features from music file."""
    y, sr = librosa.load(audio_path, mono=True)
    features = {}
    
    for col in NUMERICAL_COLS:
        if col in ["age", "music_age", "explicit"]:
            features[col] = 0  # Default values for non-audio features
        else:
            # Extract relevant audio features
            features[col] = extract_specific_audio_feature(y, sr, col)
            
    return features

def extract_specific_audio_feature(y, sr, feature_name):
    """Extract specific audio feature based on feature name."""
    feature_extractors = {
        "duration": lambda y, sr: librosa.get_duration(y=y, sr=sr),
        "acousticness": lambda y, sr: np.mean(librosa.feature.rms(y=y)),
        "key": lambda y, sr: np.argmax(librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)),
        "mode": lambda y, sr: 1 if np.mean(librosa.feature.chroma_cqt(y=y, sr=sr)) > 0 else 0,
        "speechiness": lambda y, sr: np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        "instrumentalness": lambda y, sr: np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "liveness": lambda y, sr: np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "tempo": lambda y, sr: librosa.beat.tempo(y=y, sr=sr)[0],
        "time_signature": lambda y, sr: 4,  # Default value
        "energy_loudness": lambda y, sr: np.mean(librosa.feature.rms(y=y)),
        "dance_valence": lambda y, sr: np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    }
    
    extractor = feature_extractors.get(feature_name)
    return extractor(y, sr) if extractor else 0

def prepare_user_data(user_input, audio_features=None):
    """Prepare user input data for the model."""
    user_data = pd.DataFrame({
        'user_id': [user_input.get('user_id', '0')],
        'age': [user_input['age']],
        'gender': [user_input['gender']],
        'genre': [user_input['genre']],
        'artist_name': [user_input['artist']],
        'music': [user_input['music']],
        'release_year': [user_input.get('release_year', 2024)]
    })
    
    if audio_features:
        for key, value in audio_features.items():
            user_data[key] = [value]
    else:
        for col in NUMERICAL_COLS:
            if col not in user_data.columns:
                user_data[col] = 0
    
    return user_data

