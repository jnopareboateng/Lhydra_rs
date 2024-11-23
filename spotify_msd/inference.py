import torch
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor
from model import HybridRecommender
import librosa
import sys

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

def load_model_and_preprocessor(model_path, preprocessor_path):
    """Load trained model and preprocessor"""
    import os
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    # Check if preprocessor directory exists and contains required files
    preprocessor_files = [
        "user_id_encoder.pkl",
        "gender_encoder.pkl", 
        "artist_tfidf_vectorizer.pkl",
        "genre_tfidf_vectorizer.pkl",
        "music_tfidf_vectorizer.pkl",
        "release_year_encoder.pkl",
        "scaler.pkl"
    ]
    
    for file in preprocessor_files:
        file_path = os.path.join(preprocessor_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Preprocessor file {file} not found at {preprocessor_path}. "
                "Please train the model first."
            )

    # Load preprocessor
    preprocessor = DataPreprocessor()
    try:
        preprocessor.load_preprocessors(preprocessor_path)
    except Exception as e:
        raise RuntimeError(f"Error loading preprocessors: {str(e)}")

    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridRecommender(
            num_users=len(preprocessor.user_id_encoder.classes_),
            num_genders=len(preprocessor.gender_encoder.classes_),
            num_music_items=len([col for col in preprocessor.music_tfidf_vectorizer.get_feature_names_out()]),
            num_genres=len([col for col in preprocessor.genre_tfidf_vectorizer.get_feature_names_out()]),
            num_artist_features=len([col for col in preprocessor.artist_tfidf_vectorizer.get_feature_names_out()]),
            num_numerical_features=len(NUMERICAL_COLS),
            num_release_years=len(preprocessor.release_year_encoder.categories_[0]),
            embedding_dim=EMBEDDING_DIM
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

    return model, preprocessor


def prepare_inference_features(data, preprocessor):
    """
    Prepare features for inference using the same preprocessing as training.

    Args:
        data (pd.DataFrame): Raw input data
        preprocessor (DataPreprocessor): Trainedc preprocessor

    Returns:
        dict: Processed features ready for model input
    """
    # Encode features using the preprocessor
    data_encoded =preprocessor.encode_features_transform(data)
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
    top_k=10
):
    """
    Get personalized music recommendations with robust device handling
    """
    try:
        # Force CPU if CUDA has issues
        device = torch.device("cpu")
        
        # Load model and preprocessor
        model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
        model = model.to(device)
        model.eval()

        # Prepare user data
        user_data = prepare_user_data(user_input, preprocessor)
        
        # Verify data dimensions
        if user_data.empty:
            raise ValueError("No valid user data generated")

        # Process features with dimension checks
        features = prepare_inference_features(user_data, preprocessor)
        
        # Convert to tensors with safety checks
        tensor_data = {}
        try:
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
                "release_years": torch.tensor(features['release_year_encoded'].values, dtype=torch.long).to(device)
            }
        except Exception as e:
            raise ValueError(f"Error creating tensors: {str(e)}")

        # Verify tensor dimensions
        expected_dims = model.get_expected_dimensions()
        for key, tensor in tensor_data.items():
            if tensor.shape[1] != expected_dims[key]:
                raise ValueError(f"Dimension mismatch for {key}: expected {expected_dims[key]}, got {tensor.shape[1]}")

        # Get predictions with error handling
        with torch.no_grad():
            try:
                predictions = model(
                    tensor_data["user_id"],
                    tensor_data["artist_features"],
                    tensor_data["gender_ids"],
                    tensor_data["music_features"],
                    tensor_data["genre_features"],
                    tensor_data["numerical_features"],
                    tensor_data["release_years"]
                )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print("CUDA error encountered, falling back to CPU...")
                    model = model.cpu()
                    predictions = model(
                        *[t.cpu() for t in tensor_data.values()]
                    )
                else:
                    raise

        # Process predictions safely
        predictions = predictions.cpu().numpy().flatten()
        if len(predictions) < top_k:
            raise ValueError(f"Not enough predictions generated: {len(predictions)} < {top_k}")
            
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Create recommendations with bounds checking
        if max(top_indices) >= len(user_data):
            raise IndexError("Prediction indices out of bounds")
            
        recommendations = user_data.iloc[top_indices].copy()
        recommendations["score"] = predictions[top_indices]
        
        return recommendations[["artist_name", "music", "genre", "score"]]

    except Exception as e:
        print(f"Error in personalized recommendations: {str(e)}")
        return pd.DataFrame(columns=["artist_name", "music", "genre", "score"])


def get_general_recommendations(model, preprocessor, sample_size=1000, top_k=10):
    """Generate general recommendations using trained model"""
    try:
        # Load the data with explicit path handling
        data_path = "../../data/engineered_data.csv"
        if not isinstance(data_path, (str, bytes)):
            raise TypeError("Data path must be a string")
            
        data = preprocessor.load_data(data_path)
        if data.empty:
            raise ValueError("No data loaded")
        
        # Take a random sample
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=42)
        
        # Preprocess the sampled data
        features = preprocessor.encode_features_transform(sampled_data)
        features = preprocessor.feature_engineering(features)
        
        # Prepare tensor data
        device = next(model.parameters()).device
        tensor_data = {
            "user_id": torch.tensor(features["user_id_encoded"].values, dtype=torch.long).to(device),
            "artist_features": torch.tensor(
                features[[col for col in features.columns if col.startswith("artist_tfidf_")]].values,
                dtype=torch.float32
            ).to(device),
            "gender_ids": torch.tensor(features["gender_encoded"].values, dtype=torch.long).to(device),
            "music_features": torch.tensor(
                features[[col for col in features.columns if col.startswith("music_tfidf_")]].values,
                dtype=torch.float32
            ).to(device),
            "genre_features": torch.tensor(
                features[[col for col in features.columns if col.startswith("genre_tfidf_")]].values,
                dtype=torch.float32
            ).to(device),
            "numerical_features": torch.tensor(
                features[NUMERICAL_COLS].values,
                dtype=torch.float32
            ).to(device),
            "release_years": torch.tensor(
                features["release_year_encoded"].values,
                dtype=torch.long
            ).to(device)
        }

        # Generate predictions
        model.eval()
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

        # Convert predictions to numpy and get top-k
        predictions_np = predictions.cpu().numpy().flatten()
        top_indices = np.argsort(predictions_np)[-top_k:][::-1]
        
        # Create recommendations DataFrame
        recommendations = sampled_data.iloc[top_indices].copy()
        recommendations["score"] = predictions_np[top_indices].astype(float)
        
        return recommendations[["artist_name", "music", "genre", "score"]]

    except Exception as e:
        print(f"Error generating general recommendations: {str(e)}")
        raise


def get_user_input():
    """Collect user information and preferences interactively"""
    print("\nPlease enter your information:")
    user_data = {
        "user_id": input("Enter user ID (or press Enter for new ID): ") or f"new_user_{np.random.randint(10000)}",
        "age": int(input("Enter your age: ")),
        "gender": input("Enter your gender (M/F/O): ").upper(),
        "artist": input("Enter a favorite artist: "),
        "music": input("Enter a favorite song: "),
        "genre": input("Enter preferred genre: "),
        "release_year": int(input("Enter song release year: "))
    }
    return user_data

def main():
    try:
        print("Loading model and preprocessor...")
        model, preprocessor = load_model_and_preprocessor(
            model_path="models/best_model.pth",
            preprocessor_path="models/"
        )
        print("Model and preprocessor loaded successfully!")

        while True:
            print("\nSelect an option:")
            print("1. Get general recommendations")
            print("2. Get personalized recommendations")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == "1":
                print("\nGenerating general recommendations...")
                recommendations = get_general_recommendations(model, preprocessor)
                print("\nTop 10 General Recommendations:")
                pd.set_option('display.float_format', '{:.4f}'.format)
                print(recommendations)
                
            elif choice == "2":
                # Get user input
                user_data = get_user_input()
                
                # Get personalized recommendations
                print("\nGenerating personalized recommendations...")
                recommendations = get_personalized_recommendations(
                    model_path="models/best_model.pth",
                    preprocessor_path="models/",
                    user_input=user_data
                )
                print("\nTop 10 Personalized Recommendations:")
                print(recommendations)
                
            elif choice == "3":
                print("\nExiting...")
                break
                
            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()