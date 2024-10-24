import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42, max_artist_features=3995):
        self.test_size = test_size
        self.random_state = random_state
        self.user_id_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.track_encoder = LabelEncoder()  # Changed from music_id_encoder
        self.artist_tfidf_vectorizer = TfidfVectorizer(max_features=max_artist_features)
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
    
    def encode_features(self, data):
        """
        Encode categorical features using LabelEncoder and TF-IDF Vectorizer.
        """
        # Label Encoding
        data['user_id_encoded'] = self.user_id_encoder.fit_transform(data['user_id'])
        data['gender_encoded'] = self.gender_encoder.fit_transform(data['gender'])
        data['genre_encoded'] = self.genre_encoder.fit_transform(data['genre'])
        data['track_encoded'] = self.track_encoder.fit_transform(data['music_id'])  # Changed this line
        
        # TF-IDF Encoding for 'artist_name' only
        artist_tfidf = self.artist_tfidf_vectorizer.fit_transform(data['artist_name'])
        
        # Ensure TF-IDF dimensions are consistent
        print(f"Artist TF-IDF Shape: {artist_tfidf.shape}")

        # Convert TF-IDF matrix to DataFrame
        artist_tfidf_df = pd.DataFrame(
            artist_tfidf.toarray(), 
            columns=[f'artist_tfidf_{i}' for i in range(artist_tfidf.shape[1])]
        )

        # If the number of features is less than max_artist_features, pad with zeros
        if artist_tfidf_df.shape[1] < self.artist_tfidf_vectorizer.max_features:
            for i in range(artist_tfidf_df.shape[1], self.artist_tfidf_vectorizer.max_features):
                artist_tfidf_df[f'artist_tfidf_{i}'] = 0

        # Concatenate encoded DataFrames with the original DataFrame
        data_encoded = pd.concat([
            data[['user_id_encoded', 'track_encoded', 'genre_encoded', 'gender_encoded', 'age', 'duration', 'acousticness', 
                  'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'instrumentalness', 
                  'liveness', 'valence', 'tempo', 'time_signature', 'explicit', 'plays']],
            artist_tfidf_df,
        ], axis=1)
        
        # Verify the total number of features
        total_features = data_encoded.shape[1] - 2  # Exclude 'user_id_encoded' and 'plays'
        print(f"Total features after encoding: {total_features}")
        
        print(f"Unique user IDs: {len(self.user_id_encoder.classes_)}")
        print(f"Unique artist features: {artist_tfidf.shape[1]}")
        print(f"Unique track IDs: {len(self.track_encoder.classes_)}")
        print(f"Unique genre IDs: {len(self.genre_encoder.classes_)}")
        
        return data_encoded
    
    def feature_engineering(self, data_encoded):
        """
        Perform feature scaling on numerical features.
        
        Args:
            data_encoded (pd.DataFrame): Encoded data.
        
        Returns:
            pd.DataFrame: Data with scaled numerical features.
        """
        numerical_features = ['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                              'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                              'time_signature', 'explicit']
        data_encoded[numerical_features] = self.scaler.fit_transform(data_encoded[numerical_features])
        return data_encoded
    
    def split_data(self, data_encoded, target_column='plays', val_size=0.1):
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
        features = data_encoded.drop(columns=[target_column])
        target = data_encoded[target_column]
        
        # First split to get train and temp (which will be split into validation and test)
        train_features, temp_features, train_target, temp_target = train_test_split(
            features,
            target,
            test_size=self.test_size + val_size,
            random_state=self.random_state
        )
        
        # Calculate the proportion of validation set in the temp set
        val_proportion = val_size / (self.test_size + val_size)
        
        # Split temp into validation and test sets
        val_features, test_features, val_target, test_target = train_test_split(
            temp_features,
            temp_target,
            test_size=1 - val_proportion,
            random_state=self.random_state
        )
        
        return train_features, val_features, test_features, train_target, val_target, test_target
    # def split_data(self, features, target='plays', test_size=0.2, val_size=0.1):
    #         """
    #         Split the data into training, validation, and test sets.

    #         Args:
    #             features (np.array): Feature data.
    #             target (np.array): Target data.
    #             test_size (float): Proportion of the data to include in the test split.
    #             val_size (float): Proportion of the training data to include in the validation split.

    #         Returns:
    #             tuple: Split data (train_features, val_features, test_features, train_target, val_target, test_target).
    #         """
    #         # Split into train and test
    #         train_features, test_features, train_target, test_target = train_test_split(
    #             features, target, test_size=test_size, random_state=42
    #         )
            
    #         # Further split train into train and validation
    #         train_features, val_features, train_target, val_target = train_test_split(
    #             train_features, train_target, test_size=val_size, random_state=42
    #         )
            
    #         return train_features, val_features, test_features, train_target, val_target, test_target
    
    def save_preprocessors(self, directory='models/'):
        """
        Save encoders and vectorizers to disk.
        
        Args:
            directory (str): Directory where the models will be saved.
        """
        with open(f'{directory}user_id_encoder.pkl', 'wb') as f:
            pickle.dump(self.user_id_encoder, f)
        
        with open(f'{directory}track_encoder.pkl', 'wb') as f:
            pickle.dump(self.track_encoder, f)
        
        with open(f'{directory}gender_encoder.pkl', 'wb') as f:
            pickle.dump(self.gender_encoder, f)
        
        with open(f'{directory}artist_tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.artist_tfidf_vectorizer, f)
        
        with open(f'{directory}genre_encoder.pkl', 'wb') as f:
            pickle.dump(self.genre_encoder, f)
        
        with open(f'{directory}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_preprocessors(self, directory='models/'):
        """
        Load encoders and vectorizers from disk.
        
        Args:
            directory (str): Directory where the models are saved.
        """
        with open(f'{directory}user_id_encoder.pkl', 'rb') as f:
            self.user_id_encoder = pickle.load(f)
        
        with open(f'{directory}track_encoder.pkl', 'rb') as f:
            self.track_encoder = pickle.load(f)
        
        with open(f'{directory}gender_encoder.pkl', 'rb') as f:
            self.gender_encoder = pickle.load(f)
        
        with open(f'{directory}artist_tfidf_vectorizer.pkl', 'rb') as f:
            self.artist_tfidf_vectorizer = pickle.load(f)
        
        with open(f'{directory}genre_encoder.pkl', 'rb') as f:
            self.genre_encoder = pickle.load(f)
        
        with open(f'{directory}scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
