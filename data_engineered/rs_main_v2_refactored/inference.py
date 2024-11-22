import torch
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor
from model import HybridRecommender
import librosa

# Add missing NUMERICAL_COLS definition
NUMERICAL_COLS = [
    "age",
    "duration",
    "acousticness",
    "key",
    "mode",
    "speechiness",
    "instrumentalness",
    "liveness",
    "tempo",
    "time_signature",
    "explicit",
    "music_age",
    "energy_loudness",
    "dance_valence"
]

# Add this constant at the top of the file
EMBEDDING_DIM = 32  # Must match the training configuration

def load_model_and_preprocessor(model_path, preprocessor_path="models/"):
    """
    Load the trained model and preprocessor.

    Args:
        model_path (str): Path to the saved model weights
        preprocessor_path (str): Directory containing preprocessor files

    Returns:
        tuple: (model, preprocessor)
    """
    # Load preprocessor
    preprocessor = DataPreprocessor(
        test_size=0.2,
        random_state=42,
        max_artist_features=3000,
        max_genre_features=20,
        max_music_features=5000,
    )
    preprocessor.load_preprocessors(preprocessor_path)
    
    # Initialize default values if encoders are not fitted
    default_dims = {
        "num_users": 1000,  # Default dimension for users
        "num_genders": len(preprocessor.gender_encoder.classes_),
        "num_music_items": len(preprocessor.music_tfidf_vectorizer.vocabulary_),
        "num_genres": len(preprocessor.genre_tfidf_vectorizer.vocabulary_),
        "num_artist_features": len(preprocessor.artist_tfidf_vectorizer.vocabulary_),
        "num_numerical_features": len(NUMERICAL_COLS),
        "num_release_years": len(preprocessor.release_year_encoder.categories_[0]),
    }

    # Try to get actual user dimensions if encoder is fitted
    try:
        default_dims["num_users"] = len(preprocessor.user_id_encoder.classes_)
    except AttributeError:
        print("Warning: Using default user dimension size")

    # Initialize model with dimensions matching the training configuration
    model = HybridRecommender(
        num_users=default_dims["num_users"],
        num_genders=default_dims["num_genders"],
        num_music_items=default_dims["num_music_items"],
        num_genres=default_dims["num_genres"],
        num_artist_features=default_dims["num_artist_features"],
        num_numerical_features=default_dims["num_numerical_features"],
        num_release_years=default_dims["num_release_years"],
        embedding_dim=EMBEDDING_DIM  # Use the same embedding dimension as training
    )

    # Load model weights with weights_only=True for security
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set to evaluation mode
    
    return model, preprocessor


def prepare_inference_features(data, preprocessor):
    """
    Prepare features for inference using the same preprocessing as training.

    Args:
        data (pd.DataFrame): Raw input data
        preprocessor (DataPreprocessor): Trained preprocessor

    Returns:
        dict: Processed features ready for model input
    """
    # Encode features using the preprocessor
    data_encoded = preprocessor.encode_features_transform(data)
    features = preprocessor.feature_engineering(data_encoded)

    # Ensure numerical features are floats and have correct shape
    numerical_features = data[NUMERICAL_COLS].values.astype(np.float32)
    if len(numerical_features.shape) == 1:
        numerical_features = numerical_features.reshape(-1, len(NUMERICAL_COLS))

    return features


def get_recommendations(model, user_data, preprocessor, device, top_k=10):
    """
    Generate recommendations for a user.

    Args:
        model (HybridRecommender): Trained model
        user_data (pd.DataFrame): User's data including all necessary columns
        preprocessor (DataPreprocessor): Trained preprocessor
        device (torch.device): Device to run inference on
        top_k (int): Number of recommendations to return

    Returns:
        pd.DataFrame: Top-k recommendations with scores
    """
    model.eval()

    # Prepare features
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

    # Get predictions
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

    # Convert predictions to numpy
    predictions = predictions.cpu().numpy().flatten()

    # Get top-k indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    # Create recommendations dataframe
    recommendations = user_data.iloc[top_indices].copy()
    recommendations["score"] = predictions[top_indices]

    return recommendations[["artist_name", "music", "genre", "score"]]


def extract_audio_features(audio_path):
    """Extract audio features from music file"""
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


def prepare_user_data(user_input, preprocessor, audio_features=None):
    """
    Prepare user data for inference, handling both existing and new users
    
    Args:
        user_input (dict): User input data
        preprocessor (DataPreprocessor): Trained preprocessor
        audio_features (dict, optional): Audio features if available
    
    Returns:
        pd.DataFrame: Processed user data ready for model input
    """
    # Create or get user profile
    user_profile = preprocessor.create_new_user(user_input)
    
    # Record interaction
    interaction_data = {
        'music': user_input['music'],
        'artist_name': user_input['artist'],
        'genre': user_input['genre'],
        'release_year': user_input.get('release_year', 2020),
    }
    
    # Add audio features if available
    if audio_features:
        interaction_data.update(audio_features)
    
    # Create DataFrame with interaction
    user_data = preprocessor.record_user_interaction(
        user_input['user_id'], 
        interaction_data
    )
    
    # Add user profile information
    user_data['age'] = user_profile['age']
    user_data['gender'] = user_profile['gender']
    
    return user_data


def get_personalized_recommendations(
    model_path, 
    preprocessor_path, 
    user_input, 
    audio_path=None, 
    device='cpu', 
    top_k=10
):
    """
    Get personalized music recommendations
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to preprocessor
        user_input: Dict containing user preferences
        audio_path: Optional path to music file for audio feature extraction
        device: Computing device
        top_k: Number of recommendations to return
    """
    # Load model and preprocessor with correct embedding dimension
    model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
    model = model.to(device)
    model.eval()

    # Extract audio features if audio file provided
    audio_features = None
    if audio_path:
        audio_features = extract_audio_features(audio_path)

    # Prepare user data
    user_data = prepare_user_data(user_input, preprocessor, audio_features)
    
    # Process features
    features = prepare_inference_features(user_data, preprocessor)
    
    # Convert to tensors
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
            features[[col for col in features.columns if col in NUMERICAL_COLS]].values,
            dtype=torch.float32
        ).to(device),
        "release_years": torch.tensor(features['release_year_encoded'].values, dtype=torch.long).to(device),
    }

    # Get predictions
    with torch.no_grad():
        predictions = model(
            tensor_data["user_id"],
            tensor_data["artist_features"],
            tensor_data["gender_ids"],
            tensor_data["music_features"],
            tensor_data["genre_features"],
            tensor_data["numerical_features"],
            tensor_data["release_years"],
        )

    # Get top-k recommendations
    predictions = predictions.cpu().numpy().flatten()
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    recommendations = user_data.iloc[top_indices].copy()
    recommendations["score"] = predictions[top_indices]
    
    return recommendations[["artist_name", "music", "genre", "score"]]


def main():
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(
        model_path="models/best_model.pth", preprocessor_path="models/"
    )
    model = model.to(device)

    # Load user data
    # This should include all necessary columns matching the training data
    user_data = pd.read_csv("../../data/engineered_data.csv")

    # Get recommendations
    recommendations = get_recommendations(
        model=model,
        user_data=user_data,
        preprocessor=preprocessor,
        device=device,
        top_k=10,
    )

    print("Top 10 Recommendations:")
    print(recommendations[["artist_name", "music", "score"]].to_string())

    user_input = {
        "age": 16,
        "gender": "M",
        "genre": "Christian/Gospel",
        "artist": "Blake Shelton",
        "music": "God Gave Me You",
    }
    
    personalized_recommendations = get_personalized_recommendations(
        model_path="models/best_model.pth",
        preprocessor_path="models/",
        user_input=user_input,
        audio_path=None,
        device=device,  # Add this line
        top_k=10
    )
    
    print("Top 10 Personalized Recommendations:")
    print(personalized_recommendations.to_string())


if __name__ == "__main__":
    main()
