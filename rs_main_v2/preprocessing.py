import pandas as pd
import pickle
import hashlib
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset


class DataPreprocessor:
    def __init__(
        self,
        max_artist_features,
        max_genre_features,
        max_music_features,
        test_size=0.2,
        random_state=42,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.max_artist_features = max_artist_features
        self.max_genre_features = max_genre_features
        self.max_music_features = max_music_features
        self.gender_encoder = LabelEncoder()  # Changed to LabelEncoder
        self.music_tfidf_vectorizer = TfidfVectorizer(max_features=max_music_features)
        self.artist_tfidf_vectorizer = TfidfVectorizer(max_features=max_artist_features)
        self.genre_tfidf_vectorizer = TfidfVectorizer(max_features=max_genre_features)
        self.release_year_encoder = LabelEncoder()  # Added release year encoder
        self.scaler = StandardScaler()

    def load_data(self, filepath):
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        data = pd.read_csv(filepath)
        return data

    def hash_user_id(self, user_id):
        """
        Convert user_id to 32-dimensional numeric tensor.
        Returns zero vector for null/empty values.
        """
        # Handle null/empty values
        if pd.isna(user_id) or str(user_id).strip() == "":
            return np.zeros(32)

        # Hash valid user_id
        hash_hex = hashlib.md5(str(user_id).encode()).hexdigest()

        # Convert hex string to 32 integers (2 chars per integer)
        try:
            return np.array([int(hash_hex[i : i + 2], 16) for i in range(0, 64, 2)])
        except ValueError:
            # Return zero vector if conversion fails
            return np.zeros(32)

    def encode_features(self, data):
        """Encode categorical features including release year"""
        # Hash user IDs to 32-dim vectors
        user_id_hashed = np.vstack(data["user_id"].apply(self.hash_user_id))

        # Encode gender and release year
        data["gender_encoded"] = self.gender_encoder.fit_transform(data["gender"])
        data["release_year_encoded"] = self.release_year_encoder.fit_transform(
            data["release_year"]
        )

        # Ensure release_year is included
        if "release_year" not in data.columns:
            raise ValueError("release_year column is missing from the input data")

        # TF-IDF Encoding
        artist_tfidf = self.artist_tfidf_vectorizer.fit_transform(data["artist_name"])
        genre_tfidf = self.genre_tfidf_vectorizer.fit_transform(data["genre"])
        music_tfidf = self.music_tfidf_vectorizer.fit_transform(data["music"])

        # Get actual feature names from vectorizers
        artist_feature_names = [
            f"artist_tfidf_{i}" for i in range(artist_tfidf.shape[1])
        ]
        genre_feature_names = [f"genre_tfidf_{i}" for i in range(genre_tfidf.shape[1])]
        music_feature_names = [f"music_tfidf_{i}" for i in range(music_tfidf.shape[1])]

        # Create DataFrames with actual dimensions
        artist_tfidf_df = pd.DataFrame(
            artist_tfidf.toarray(), columns=artist_feature_names
        )
        genre_tfidf_df = pd.DataFrame(
            genre_tfidf.toarray(), columns=genre_feature_names
        )
        music_tfidf_df = pd.DataFrame(
            music_tfidf.toarray(), columns=music_feature_names
        )

        # Combine all features in the expected order
        numerical_features = [
            "age",
            "duration",
            "acousticness",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "time_signature",
            "explicit",
        ]

        data_encoded = pd.DataFrame(
            {
                "user_id_hashed": list(user_id_hashed),
                "gender_encoded": data["gender_encoded"],
                "release_year_encoded": data["release_year_encoded"],
                "release_year": data["release_year"],  # Ensure this is included
            }
        )

        data_encoded = pd.concat(
            [
                data_encoded,
                music_tfidf_df,
                artist_tfidf_df,
                genre_tfidf_df,
                data[numerical_features],
            ],
            axis=1,
        )

        return data_encoded

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering with new target variables."""
        # Original numerical features
        numerical_features = [
            "age",
            "duration",
            "acousticness",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "time_signature",
            "explicit",
        ]

        # Store plays separately before scaling
        target = None
        if "plays" in data.columns:
            target = data["plays"]
            # Normalize plays to [0,1] range
            target = (target - target.min()) / (target.max() - target.min())

        # Scale numerical features
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])

        # Add normalized plays back
        if target is not None:
            data["plays"] = target

        # Add new target variables
        data["genre_diversity"] = self.calculate_genre_diversity(data)
        data["novelty"] = self.calculate_novelty_score(data)
        data["popularity"] = self.calculate_popularity_score(data)

        return data

    def calculate_genre_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate genre diversity score for each item."""
        genre_features = data[
            [col for col in data.columns if col.startswith("genre_tfidf_")]
        ]
        return genre_features.apply(lambda x: (x > 0).sum() / len(x), axis=1)

    def calculate_novelty_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate novelty score based on artist and song popularity."""
        # Normalize play counts
        play_counts = data["plays"]
        normalized_plays = (play_counts - play_counts.min()) / (
            play_counts.max() - play_counts.min()
        )

        # Inverse popularity = novelty
        return 1 - normalized_plays

    def calculate_popularity_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate popularity score based on play counts and recency."""
        # Ensure release_year is available
        if "release_year" not in data.columns:
            raise ValueError("release_year column is missing from the data")

        # Normalize play counts
        play_counts = data["plays"]
        normalized_plays = (play_counts - play_counts.min()) / (
            play_counts.max() - play_counts.min()
        )

        # Calculate recency score (assuming release_year is available)
        current_year = data["release_year"].max()
        years_old = current_year - data["release_year"]
        recency_score = 1 / (1 + np.exp(-years_old))  # Sigmoid transformation

        # Combine plays and recency
        return 0.7 * normalized_plays + 0.3 * recency_score

    def split_data(self, data_encoded, target_column="plays", val_size=0.1):
        """
        Split data into training, validation, and testing sets.

        Args:
            data_encoded (pd.DataFrame): Encoded and scaled data.
            target_column (str): Name of the target column.
            val_size (float): Proportion of the data to include in the validation set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            Train features, validation features, test features, train target, validation target, test target.
        """
        required_columns = ["user_id_hashed", "gender_encoded", "release_year_encoded"]

        missing = [col for col in required_columns if col not in data_encoded.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        features = data_encoded.drop(columns=[target_column])
        target = data_encoded[target_column]

        # First split to get train and temp (which will be split into validation and test)
        train_features, temp_features, train_target, temp_target = train_test_split(
            features,
            target,
            test_size=self.test_size + val_size,
            random_state=self.random_state,
        )

        # Calculate the proportion of validation set in the temp set
        val_proportion = val_size / (self.test_size + val_size)

        # Split temp into validation and test sets
        val_features, test_features, val_target, test_target = train_test_split(
            temp_features,
            temp_target,
            test_size=1 - val_proportion,
            random_state=self.random_state,
        )

        return (
            train_features,
            val_features,
            test_features,
            train_target,
            val_target,
            test_target,
        )

    def save_preprocessors(self, directory="models/"):
        """Save all preprocessors including release year encoder"""
        preprocessors = {
            "gender_encoder": self.gender_encoder,
            "artist_tfidf_vectorizer": self.artist_tfidf_vectorizer,
            "genre_tfidf_vectorizer": self.genre_tfidf_vectorizer,
            "music_tfidf_vectorizer": self.music_tfidf_vectorizer,
            "release_year_encoder": self.release_year_encoder,
            "scaler": self.scaler,
        }

        for name, preprocessor in preprocessors.items():
            with open(f"{directory}{name}.pkl", "wb") as f:
                pickle.dump(preprocessor, f)

    def load_preprocessors(self, directory="models/"):
        """Load all preprocessors including release year encoder"""
        preprocessors = {
            "gender_encoder": "gender_encoder.pkl",
            "artist_tfidf_vectorizer": "artist_tfidf_vectorizer.pkl",
            "genre_tfidf_vectorizer": "genre_tfidf_vectorizer.pkl",
            "music_tfidf_vectorizer": "music_tfidf_vectorizer.pkl",
            "release_year_encoder": "release_year_encoder.pkl",
            "scaler": "scaler.pkl",
        }

        for attr, filename in preprocessors.items():
            with open(f"{directory}{filename}", "rb") as f:
                setattr(self, attr, pickle.load(f))

    def load_and_process_data(self):
        """Load and preprocess data returning train, validation and test sets"""
        # TODO: Replace with actual data loading
        # For now creating dummy data for testing
        n_samples = 10000
        
        # Generate dummy data
        data = {
            "user_id_hashed": torch.randn(n_samples, 32),
            "artist_features": torch.randn(n_samples, self.max_artist_features),
            "gender_ids": torch.randint(0, 2, (n_samples,)),
            "music_features": torch.randn(n_samples, self.max_music_features),
            "genre_features": torch.randn(n_samples, self.max_genre_features),
            "numerical_features": torch.randn(n_samples, 5),
            "release_years": torch.randint(0, 100, (n_samples,)),
            "target": torch.randn(n_samples)
        }

        # Split indices
        indices = np.arange(n_samples)
        train_indices, test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )
        val_indices, test_indices = train_test_split(
            test_indices, test_size=0.5, random_state=self.random_state
        )

        # Create datasets
        def create_dataset(indices):
            return TensorDataset(
                data["user_id_hashed"][indices],
                data["artist_features"][indices],
                data["gender_ids"][indices],
                data["music_features"][indices],
                data["genre_features"][indices],
                data["numerical_features"][indices],
                data["release_years"][indices],
                data["target"][indices]
            )

        train_data = create_dataset(train_indices)
        val_data = create_dataset(val_indices)
        test_data = create_dataset(test_indices)

        return train_data, val_data, test_data

    def load_real_data(self, data_path: str):
        """
        Load real data from files
        To be implemented based on your actual data format
        """
        pass