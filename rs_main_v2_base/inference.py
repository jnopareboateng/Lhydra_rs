import torch
import pandas as pd
from preprocessing import DataPreprocessor
from model import HybridRecommender
import pickle

def make_inference(model, user_id, artist_features, genre_id, numerical_features, 
                   user_encoder, track_encoder, genre_encoder, device):
    """
    Perform inference to predict plays for a specific user-item pair.
    
    Args:
        model (nn.Module): Trained recommender model.
        user_id (str): The user ID.
        artist_features (pd.Series or np.array): Feature vector for the artist.
        track_id (str): The track ID.
        genre_id (str): The genre ID.
        numerical_features (pd.Series or np.array): Feature vector for the numerical features.
        user_encoder (LabelEncoder): Fitted LabelEncoder for user IDs.
        track_encoder (LabelEncoder): Fitted LabelEncoder for track IDs.
        genre_encoder (LabelEncoder): Fitted LabelEncoder for genre IDs.
        device (torch.device): Device where the model is loaded.
    
    Returns:
        float: Predicted number of plays.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode IDs, handling OOV by assigning a special index
    def encode_id(encoder, id_value, default_index):
        try:
            return torch.tensor(encoder.transform([id_value]), dtype=torch.long).to(device)
        except ValueError:
            return torch.tensor([default_index], dtype=torch.long).to(device)
    
    # Assign default indices for OOV
    default_user_idx = len(user_encoder.classes_)
    # default_track_idx = len(track_encoder.classes_)
    default_genre_idx = len(genre_encoder.classes_)
    default_
    
    user_id_encoded = encode_id(user_encoder, user_id, default_user_idx)
    track_id_encoded = encode_id(track_encoder, track_id, default_track_idx)
    genre_id_encoded = encode_id(genre_encoder, genre_id, default_genre_idx)
    
    # Process audio features
    if isinstance(artist_features, pd.Series):
        artist_features = artist_features.values
    artist_features_tensor = torch.tensor(artist_features, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(user_id_encoded, artist_features_tensor, track_id_encoded, genre_id_encoded, numerical_features)
    return prediction.cpu().numpy()[0]

def map_music_ids_to_info(music_ids, music_id_to_info):
    """
    Map music IDs to their corresponding names and artists.

    Args:
        music_ids (List[int]): List of music IDs.
        music_id_to_info (dict): Dictionary mapping music IDs to (music_name, artist_name).

    Returns:
        List[Tuple[str, str]]: List of tuples containing music names and artist names.
    """
    return [music_id_to_info.get(music_id, ("Unknown", "Unknown")) for music_id in music_ids]


def get_recommendations(model, user_id, data_encoded, user_id_encoder, track_encoder, genre_encoder, 
                        music_id_to_info, device, top_k=10):
    """
    Generate top-k recommendations for a given user.

    Args:
        model (nn.Module): Trained recommender model.
        user_id (str): The user ID for whom to generate recommendations.
        data_encoded (pd.DataFrame): The preprocessed and encoded dataset.
        user_id_encoder (LabelEncoder): Fitted LabelEncoder for user IDs.
        track_encoder (LabelEncoder): Fitted LabelEncoder for track IDs.
        genre_encoder (LabelEncoder): Fitted LabelEncoder for genre IDs.
        music_id_to_info (dict): Mapping from music IDs to their information.
        device (torch.device): Device where the model is loaded.
        top_k (int): Number of top recommendations to return.

    Returns:
        List[Tuple[str, str]]: List of recommended music titles and their artists.
    """
    model.eval()  # Set model to evaluation mode

    # Encode the user ID, handling OOV
    try:
        user_id_encoded = user_id_encoder.transform([user_id])[0]
    except ValueError:
        print(f"User ID {user_id} not found in encoder. Using default OOV index.")
        user_id_encoded = len(user_id_encoder.classes_)

    # Extract user features
    user_data = data_encoded[data_encoded['user_id_encoded'] == user_id_encoded]
    if user_data.empty:
        print(f"No data found for user ID {user_id}. Using average user features.")
        user_features = data_encoded.drop(columns=['user_id_encoded', 'plays']).mean().values
    else:
        user_features = user_data.drop(columns=['user_id_encoded', 'plays']).iloc[0].values

    user_features_tensor = torch.tensor(user_features, dtype=torch.float).unsqueeze(0).to(device)

    # Prepare all items
    item_ids = data_encoded['track_encoded'].unique()
    item_ids_encoded = []
    numerical_features_list = []
    artist_features_list = []
    genre_ids = []

    for item_id in item_ids:
        # Retrieve encoded IDs and features
        item_data = data_encoded[data_encoded['track_encoded'] == item_id].iloc[0]
        
        artist_features = item_data[[col for col in data_encoded.columns if col.startswith('artist_tfidf_')]].values
        genre_encoded = item_data['genre_encoded']
        numerical_features = item_data[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                        'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                                        'time_signature', 'explicit']].values

        item_ids_encoded.append(item_id)
        artist_features_list.append(artist_features)
        genre_ids.append(genre_encoded)
        numerical_features_list.append(numerical_features)

    # Convert lists to tensors
    artist_features_tensor = torch.tensor(artist_features_list, dtype=torch.float).to(device)
    track_ids_tensor = torch.tensor(item_ids_encoded, dtype=torch.long).to(device)
    genre_ids_tensor = torch.tensor(genre_ids, dtype=torch.long).to(device)
    numerical_features_tensor = torch.tensor(numerical_features_list, dtype=torch.float).to(device)
    user_ids_tensor = torch.tensor([user_id_encoded] * len(item_ids_encoded), dtype=torch.long).to(device)

    # Perform inference
    with torch.no_grad():
        scores = model(user_ids_tensor, artist_features_tensor, track_ids_tensor, genre_ids_tensor, numerical_features_tensor)

    # Get top-k scores and corresponding item IDs
    top_scores, top_indices = torch.topk(scores, top_k)
    top_item_ids = [item_ids_encoded[idx] for idx in top_indices.cpu().numpy()]

    # Map to music information
    recommended_items = map_music_ids_to_info(top_item_ids, music_id_to_info)

    return recommended_items
