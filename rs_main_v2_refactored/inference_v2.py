import torch
import pandas as pd
from model import HybridRecommender
from preprocessing import DataPreprocessor
from train import NUMERICAL_COLS

# Step 1: Load the trained model and preprocessors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the preprocessor and load the saved preprocessors
preprocessor = DataPreprocessor()
preprocessor.load_preprocessors(directory="models/")

# Load the trained model parameters
model = HybridRecommender(
    num_genders=len(preprocessor.gender_encoder.classes_),
    num_music_items=preprocessor.music_tfidf_vectorizer.max_features,
    num_genres=preprocessor.genre_tfidf_vectorizer.max_features,
    num_artist_features=preprocessor.artist_tfidf_vectorizer.max_features,
    num_numerical_features=len(preprocessor.scaler.mean_),
    num_release_years=len(preprocessor.release_year_encoder.categories_[0]),
    embedding_dim=64,
)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Step 2: Prepare the user's input data
user_data = pd.DataFrame(
    [
        {
            "user_id": "new_user_id",
            "gender": "M",
            "age": 25,
            "music": "pop rock",
            "artist_name": "Imagine Dragons",
            "genre": "Alternative/Indie",
            "release_year": 2018,
            "duration": 210,
            "acousticness": 0.2,
            "danceability": 0.8,
            "energy": 0.9,
            "key": 5,
            "loudness": -5.0,
            "mode": 1,
            "speechiness": 0.05,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.7,
            "tempo": 120.0,
            "time_signature": 4,
            "explicit": 0,
        }
    ]
)

# Step 3: Encode and preprocess the input data
data_encoded = preprocessor.transform(user_data)
data_encoded = preprocessor.feature_engineering(data_encoded)

# Verify that the number of music features matches the expected input size
music_feature_cols = [
    col for col in data_encoded.columns if col.startswith("music_tfidf_")
]
print(f"Number of music features: {len(music_feature_cols)}")  # Should be 5784

# If the number is less than expected, this indicates a mismatch

# Step 4: Convert data to tensors
with torch.no_grad():
    user_id_hashed = torch.tensor(
        data_encoded["user_id_hashed"].tolist(), dtype=torch.float32
    ).to(device)
    gender_id = (
        torch.tensor(data_encoded["gender_encoded"].values, dtype=torch.long)
        .unsqueeze(1)
        .to(device)
    )
    release_year = (
        torch.tensor(data_encoded["release_year_encoded"].values, dtype=torch.long)
        .unsqueeze(1)
        .to(device)
    )
    music_features = torch.tensor(
        data_encoded.filter(regex="^music_tfidf_").values, dtype=torch.float32
    ).to(device)
    # Ensure music_features has the correct shape
    print(f"music_features shape: {music_features.shape}")  # Should output: [1, 5784]

    artist_features = torch.tensor(
        data_encoded.filter(regex="^artist_tfidf_").values, dtype=torch.float32
    ).to(device)
    genre_features = torch.tensor(
        data_encoded.filter(regex="^genre_tfidf_").values, dtype=torch.float32
    ).to(device)
    numerical_features = torch.tensor(
        data_encoded[NUMERICAL_COLS].values, dtype=torch.float32
    ).to(device)

# Step 5: Make predictions using the model
with torch.no_grad():
    prediction = model(
        user_id_hashed=user_id_hashed,
        artist_features=artist_features,
        gender_id=gender_id,
        music_features=music_features,
        genre_features=genre_features,
        numerical_features=numerical_features,
        release_year=release_year,
    )

    # Output the prediction
    print("Predicted score:", prediction.item())
