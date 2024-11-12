import torch
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor
from model import HybridRecommender

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
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessors(preprocessor_path)
    
    # Initialize model with correct dimensions
    model = HybridRecommender(
        num_genders=len(preprocessor.gender_encoder.classes_),
        num_music_items=preprocessor.music_tfidf_vectorizer.max_features,
        num_genres=preprocessor.genre_tfidf_vectorizer.max_features,
        num_artist_features=preprocessor.artist_tfidf_vectorizer.max_features,
        num_numerical_features=15,  # Fixed number of numerical features
        num_release_years=len(preprocessor.release_year_encoder.classes_),
        embedding_dim=64,
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
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
    data_encoded = preprocessor.encode_features(data)
    features = preprocessor.feature_engineering(data_encoded)
    
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
    
    # Define numerical columns to match training
    numerical_cols = [
        "age", "duration", "acousticness", "danceability", "energy",
        "key", "loudness", "mode", "speechiness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature", "explicit"
    ]
    
    # Convert features to tensors
    tensor_data = {
        # 'user_id_hashed': torch.tensor(features['user_id_hashed'].values, dtype=torch.float).to(device),  # Changed from user_id to user_id_hashed and dtype to float
        # 'user_id_hashed': torch.tensor(np.stack(features["user_id_hashed"].tolist()), dtype=torch.float).to(device),
        # 'gender_ids': torch.tensor(features['gender_encoded'].values, dtype=torch.long).to(device),
        # 'genre_features': torch.tensor(features[[col for col in features.columns if col.startswith('genre_tfidf_')]].values, dtype=torch.float).to(device),
        # 'artist_features': torch.tensor(features[[col for col in features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device),
        # 'music_features': torch.tensor(features[[col for col in features.columns if col.startswith('music_tfidf_')]].values, dtype=torch.float).to(device),
        # 'numerical_features': torch.tensor(features[numerical_cols].values, dtype=torch.float).to(device),
        # 'release_years': torch.tensor(features['release_year_encoded'].values, dtype=torch.long).to(device),
        'user_id_hashed': torch.tensor(np.stack(features["user_id_hashed"].tolist()), dtype=torch.float).to(device),
        'gender_ids': torch.tensor(features["gender_encoded"].values, dtype=torch.long).to(device),
        'genre_features': torch.tensor(features[[col for col in features.columns if col.startswith("genre_tfidf_")]].values, dtype=torch.float).to(device),
        'artist_features': torch.tensor(features[[col for col in features.columns if col.startswith("artist_tfidf_")]].values, dtype=torch.float).to(device),
        'music_features': torch.tensor(features[[col for col in features.columns if col.startswith("music_tfidf_")]].values, dtype=torch.float).to(device),
        'numerical_features': torch.tensor(features[numerical_cols].values, dtype=torch.float).to(device),
        'release_years': torch.tensor(features["release_year_encoded"].values, dtype=torch.long).to(device),
        # 'target': torch.tensor(target.values, dtype=torch.float).unsqueeze(1).to(device)
    }
    
    # Get predictions
    with torch.no_grad():
        predictions = model(
            tensor_data['user_id_hashed'],
            tensor_data['gender_ids'],
            tensor_data['genre_features'],
            tensor_data['artist_features'],
            tensor_data['music_features'],
            tensor_data['numerical_features'],
            tensor_data['release_years'],
        )
    
    # Convert predictions to numpy
    predictions = predictions.cpu().numpy().flatten()
    
    # Get top-k indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    # Create recommendations dataframe
    recommendations = user_data.iloc[top_indices].copy()
    recommendations['score'] = predictions[top_indices]
    
    return recommendations

def main():
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(
        model_path="models/best_model.pth",
        preprocessor_path="models/"
    )
    model = model.to(device)
    
    # Load user data
    # This should include all necessary columns matching the training data
    user_data = pd.read_csv("./data/cleaned_modv2.csv")
    
    # Get recommendations
    recommendations = get_recommendations(
        model=model,
        user_data=user_data,
        preprocessor=preprocessor,
        device=device,
        top_k=10
    )
    
    print("Top 10 Recommendations:")
    print(recommendations[['artist_name', 'music', 'score']].to_string())

if __name__ == "__main__":
    main()
