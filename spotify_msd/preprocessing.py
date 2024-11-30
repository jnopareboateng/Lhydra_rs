import pandas as pd
import pickle

# import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPreprocessor:
    NUMERICAL_COLS = [
        "age",
        "gender",
        "duration_ms",
        "loudness",
        "instrumentalness",
        "liveness",
        "tempo",
        "energy",
        "danceability",
        "valence",
        "year",
    ]

CATEGORICAL_COLS = [
    "gender",
    "genre",
    "artist",
    "track_id"
]
    def __init__(
        self,
        test_size=0.2,
        random_state=42,
        max_artist_features=3000,
        max_genre_features=20,
        max_music_features=5000,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.gender_encoder = LabelEncoder()
        self.user_id_encoder = LabelEncoder()
        self.artist_tfidf_vectorizer = TfidfVectorizer(max_features=max_artist_features)
        self.genre_tfidf_vectorizer = TfidfVectorizer(max_features=max_genre_features)
        self.music_tfidf_vectorizer = TfidfVectorizer(max_features=max_music_features)
        self.release_year_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )  # Added release year encoder
        self.scaler = StandardScaler()
        self.unknown_user_id = "unknown_user"
        self.default_user_data = {
            "age": 25,  # Default age
            "gender": "Male",  # Default
            "playcount": 0,
        }

    def load_data(self, filepath):
        """
        Load data from CSV with explicit dtypes
        """
        dtype_dict = {
            "age": np.float32,
            "duration_ms": np.float32,
            "acousticness": np.float32,
            "key": np.float32,
            "mode": np.float32,
            "speechiness": np.float32,
            "instrumentalness": np.float32,
            "liveness": np.float32,
            "tempo": np.float32,
            "time_signature": np.float32,
            "explicit": np.float32,
            "music_age": np.float32,
            "energy_loudness": np.float32,
            "dance_valence": np.float32,
            "playcount": np.float32,
            "user_id": str,  # Will be encoded later
            "gender": str,  # Will be encoded later
            "year": np.int32,
        }

        data = pd.read_csv(filepath, dtype=dtype_dict)
        return data

    def fit(self, data):
        """Fit all preprocessors on training data"""
        self.gender_encoder.fit(data["gender"])
        self.release_year_encoder.fit(data["year"])
        self.artist_tfidf_vectorizer.fit(data["artist"])
        self.genre_tfidf_vectorizer.fit(data["genre"])
        self.music_tfidf_vectorizer.fit(data["name"])
        self.scaler.fit(data[self.NUMERICAL_COLS])
        return self

    def transform(self, data, is_training=False):
        """Transform data using fitted preprocessors"""
        transform_fn = lambda enc, x: (
            enc.fit_transform(x) if is_training else enc.transform(x)
        )

        data_encoded = pd.DataFrame(
            {
                # "user_id_hashed": np.vstack(data["user_id"].apply(self.hash_user_id)),
                "gender_encoded": transform_fn(self.gender_encoder, data["gender"]),
                "release_year_encoded": transform_fn(
                    self.release_year_encoder, data["year"]
                ),
            }
        )

        # Add other transformations...
        return data_encoded

    def encode_features_train(self, data):
        # Encode user_id
        data["user_id_encoded"] = self.user_id_encoder.fit_transform(data["user_id"])
        # Ensure unique user count
        self.num_users = len(self.user_id_encoder.classes_)

        # Encode gender
        data["gender_encoded"] = self.gender_encoder.fit_transform(data["gender"])

        # Reshape release_year to 2D array and encode
        release_year_2d = data["year"].values.reshape(-1, 1)
        data["release_year_encoded"] = self.release_year_encoder.fit_transform(
            release_year_2d
        ).ravel()

        # Rest of the method remains the same...
        # TF-IDF Encoding
        artist_tfidf = self.artist_tfidf_vectorizer.fit_transform(data["artist"])
        genre_tfidf = self.genre_tfidf_vectorizer.fit_transform(data["genre"])
        music_tfidf = self.music_tfidf_vectorizer.fit_transform(data["name"])

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
        "gender",
        "duration_ms",
        "loudness",
        "instrumentalness",
        "liveness",
        "tempo",
        "energy",
        "danceability",
        "valence",
        "year",

        ]

        data_encoded = pd.DataFrame(
            {
                "user_id_encoded": data["user_id_encoded"],
                "gender_encoded": data["gender_encoded"],
                "release_year_encoded": data["release_year_encoded"],
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

        # Verify release_year exists
        if "year" not in data.columns:
            raise ValueError("release_year column missing from input data")

        # Add release year check after encoding
        if "release_year_encoded" not in data_encoded.columns:
            raise ValueError("release_year_encoded not properly created")

        return data_encoded

    def encode_features_transform(self, data):
        # Encode user_id
        data["user_id_encoded"] = self.user_id_encoder.transform(data["user_id"])

        # Encode gender
        data["gender_encoded"] = self.gender_encoder.transform(data["gender"])

        # Reshape release_year to 2D array and encode
        release_year_2d = data["year"].values.reshape(-1, 1)
        data["release_year_encoded"] = self.release_year_encoder.transform(
            release_year_2d
        ).ravel()

        # TF-IDF Encoding
        artist_tfidf = self.artist_tfidf_vectorizer.transform(data["artist"])
        genre_tfidf = self.genre_tfidf_vectorizer.transform(data["genre"])
        music_tfidf = self.music_tfidf_vectorizer.transform(data["name"])

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
        data_encoded = pd.DataFrame(
            {
                "user_id_encoded": data["user_id_encoded"],
                "gender_encoded": data["gender_encoded"],
                "release_year_encoded": data["release_year_encoded"],
            }
        )

        data_encoded = pd.concat(
            [
                data_encoded,
                music_tfidf_df,
                artist_tfidf_df,
                genre_tfidf_df,
                data[self.NUMERICAL_COLS],
            ],
            axis=1,
        )

        return data_encoded

    def feature_engineering(self, data_encoded):
        """
        Modified to handle already preprocessed numerical features and ensure numeric types
        """
        # Convert numerical columns to float32
        for col in self.NUMERICAL_COLS:
            if col not in data_encoded.columns:
                raise ValueError(f"Missing numerical column: {col}")
            # Convert to numeric, replacing any non-numeric values with NaN
            data_encoded[col] = pd.to_numeric(data_encoded[col], errors="coerce")
            # Fill NaN values with 0 or another appropriate value
            data_encoded[col] = data_encoded[col].fillna(0).astype(np.float32)

        return data_encoded

    def split_data(self, data_encoded, target_column="playcount", val_size=0.1):
        features = data_encoded.drop(columns=[target_column])
        target = data_encoded[target_column]

        # Handle duplicate values in binning
        try:
            bins = pd.qcut(target, q=10, labels=False, duplicates="drop")
        except ValueError:
            # Fallback if too many duplicates
            bins = pd.cut(target, bins=10, labels=False)

        # Split with stratification
        train_features, temp_features, train_target, temp_target = train_test_split(
            features,
            target,
            test_size=self.test_size + val_size,
            random_state=self.random_state,
            stratify=bins,
        )

        # Further split temp into validation and test
        val_features, test_features, val_target, test_target = train_test_split(
            temp_features,
            temp_target,
            test_size=self.test_size / (self.test_size + val_size),
            random_state=self.random_state,
            stratify=pd.qcut(temp_target, q=5, labels=False, duplicates="drop"),
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
        preprocessors = {
            "user_id_encoder": self.user_id_encoder,
            "gender_encoder": self.gender_encoder,
            "artist_tfidf_vectorizer": self.artist_tfidf_vectorizer,
            "genre_tfidf_vectorizer": self.genre_tfidf_vectorizer,
            "music_tfidf_vectorizer": self.music_tfidf_vectorizer,
            "release_year_encoder": self.release_year_encoder,
            "scaler": self.scaler,
        }
        for name, preprocessor in preprocessors.items():
            with open(f"{directory}/{name}.pkl", "wb") as f:
                pickle.dump(preprocessor, f)

    def load_preprocessors(self, directory="models/"):
        preprocessors = [
            "user_id_encoder",
            "gender_encoder",
            "artist_tfidf_vectorizer",
            "genre_tfidf_vectorizer",
            "music_tfidf_vectorizer",
            "release_year_encoder",
            "scaler",
        ]
        for name in preprocessors:
            with open(f"{directory}/{name}.pkl", "rb") as f:
                setattr(self, name, pickle.load(f))

    def create_new_user(self, user_data):
        """
        Create a new user profile and return encoded user data

        Args:
            user_data (dict): Dictionary containing user information
                Required keys: user_id
                Optional keys: age, gender
        Returns:
            str: Encoded user_id
        """
        # Validate required fields
        if "user_id" not in user_data:
            raise ValueError("user_id is required for new user creation")

        # Merge with default values
        user_profile = {**self.default_user_data, **user_data}

        # Add user to encoder if not exists
        if not hasattr(self.user_id_encoder, "classes_"):
            self.user_id_encoder.fit([self.unknown_user_id])

        if user_data["user_id"] not in self.user_id_encoder.classes_:
            new_classes = np.append(
                self.user_id_encoder.classes_, [user_data["user_id"]]
            )
            self.user_id_encoder.classes_ = new_classes

        return user_profile

    def record_user_interaction(self, user_id, interaction_data):
        """
        Record a user interaction with the system

        Args:
            user_id (str): User identifier
            interaction_data (dict): Dictionary containing interaction information
                Required keys: music, artist, genre
                Optional keys: playcount, duration_ms, release_year
        Returns:
            pd.DataFrame: Processed interaction data ready for model input
        """
        # Validate required fields
        required_fields = ["name", "artist", "genre"]
        if not all(field in interaction_data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        # Create DataFrame with single interaction
        interaction_df = pd.DataFrame(
            {
                "user_id": [user_id],
                "music": [interaction_data["name"]],
                "artist": [interaction_data["artist"]],
                "genre": [interaction_data["genre"]],
                "playcount": [interaction_data.get("playcount", 1)],
                "duration_ms": [interaction_data.get("duration_ms", 0)],
                "year": [interaction_data.get("year", 2023)],
            }
        )

        # Add default values for required numerical columns
        for col in self.NUMERICAL_COLS:
            if col not in interaction_df.columns:
                interaction_df[col] = 0.0

        return interaction_df

    def update_encoders_with_new_data(self, new_data):
        """
        Update encoders with new data without forgetting existing encodings

        Args:
            new_data (pd.DataFrame): New data to incorporate
        """
        for column, encoder in [
            ("user_id", self.user_id_encoder),
            ("gender", self.gender_encoder),
        ]:
            if column in new_data.columns:
                current_classes = encoder.classes_
                new_classes = new_data[column].unique()
                combined_classes = np.unique(
                    np.concatenate([current_classes, new_classes])
                )
                encoder.classes_ = combined_classes


# Add this check in your preprocessing pipeline
def validate_user_ids(data):
    if not np.issubdtype(data["user_id"].dtype, np.number):
        raise ValueError("user_id column must contain numeric values")
