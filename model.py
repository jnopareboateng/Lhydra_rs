import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df.drop(columns=["release_date", "release_year"], inplace=True)

df.columns
# Index(['user_id', 'age', 'gender', 'music', 'artist_name', 'featured_artists',
#        'genre', 'plays', 'duration', 'music_id', 'id_artists', 'acousticness',
#        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
#        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
#        'explicit', 'rating', 'age_group'],
#       dtype='object')

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

df["time_signature"].unique()
# array([4., 3., 1., 5., 0.])

# Normalize numerical features
numerical_features = [
    "age",
    "plays",
    "duration",
    "key",
    "mode",
    "explicit",
    "rating",
    "acousticness",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

scaler = MinMaxScaler()
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])


# Convert pandas DataFrame to TensorFlow Dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=128):
    dataframe = df.copy()
    labels = dataframe.pop(
        "plays"
    )  # Replace 'rating' with your actual label column if different
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 128
train_dataset = df_to_dataset(train_df, batch_size=batch_size)
test_dataset = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)

# Create item dataset
item_df = (
    df[
        [
            "music_id",
            "genre",
            "acousticness",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)
item_dataset = tf.data.Dataset.from_tensor_slices(dict(item_df))

def preprocess(features: dict, label: tf.Tensor = None) -> dict:
    """
    Preprocesses the input features and label.

    Args:
    features (dict): A dictionary containing the input features.
    label (tf.Tensor, optional): The corresponding label. Defaults to None.

    Returns:
    dict: A dictionary containing the processed features and label (if provided).
    """
    try:
        processed_features = {
            "user_id": features["user_id"],
            "age": features["age"],
            "age_group": features["age_group"],
            "gender": features["gender"],
            "music_id": features["music_id"],
            "genre": features["genre"],
            "artist_name": features.get("artist_name", ""),
            "audio_features": tf.concat(
                [
                    tf.reshape(features["acousticness"], (-1, 1)),
                    tf.reshape(features["danceability"], (-1, 1)),
                    tf.reshape(features["energy"], (-1, 1)),
                    tf.reshape(features["loudness"], (-1, 1)),
                    tf.reshape(features["speechiness"], (-1, 1)),
                    tf.reshape(features["instrumentalness"], (-1, 1)),
                    tf.reshape(features["liveness"], (-1, 1)),
                    tf.reshape(features["valence"], (-1, 1)),
                    tf.reshape(features["tempo"], (-1, 1)),
                ],
                axis=1,
            ),
        }
        if label is not None:
            processed_features['label'] = label
        return processed_features
    except KeyError as e:
        print(f"Error: Missing feature - {e}")
        return None


train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

num_users = df["user_id"].nunique()
num_age_groups = df["age_group"].nunique()
num_music = df["music_id"].nunique()
num_genders = df["gender"].nunique()
num_genres = df["genre"].nunique()
num_artists = df["artist_name"].nunique()
num_age = df["age"].nunique()

print(
    f'num_users: {num_users}\n{"="*40}\nnum_age_groups: {num_age_groups}\n{"="*40}\nnum_music: {num_music}\n{"="*40}\nnum_genders: {num_genders}\n{"="*40}\nnum_genres:{num_genres}\n{"="*40}\nnum_artists: {num_artists}\n{"="*40}\nnum_age: {num_age}'
)

## Output
# num_users: 9741
# ========================================
# num_age_groups: 5
# ========================================
# num_music: 11528
# ========================================
# num_genders: 2
# ========================================
# num_genres:21
# ========================================
# num_artists: 5085
# ========================================
# num_age: 68

embedding_dim_user = 64
embedding_dim_age_group = 8
embedding_dim_gender = 2
embedding_dim_music = 128
embedding_dim_genre = 16
embedding_dim_artist = 64
embedding_dim_audio_features = len(audio_features)

#  Define user and item models with embeddings
class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=num_users, output_dim=embedding_dim_user
        )
        self.age_embedding = tf.keras.layers.Embedding(
            input_dim=num_age_groups, output_dim=embedding_dim_age_group
        )
        self.gender_embedding = tf.keras.layers.Embedding(
            input_dim=num_genders, output_dim=embedding_dim_gender
        )
        # self.genre_embedding = tf.keras.layers.Embedding(input_dim=num_genres, output_dim=embedding_dim)
        self.favorite_music_embedding = tf.keras.layers.Embedding(
            input_dim=num_music, output_dim=embedding_dim_music
        )
        self.favorite_artist_embedding = tf.keras.layers.Embedding(
            input_dim=num_artists, output_dim=embedding_dim_artist
        )

    def call(self, user_id, age_group, gender, music_id, artist_name):
        return tf.concat(
            [
                self.user_embedding(user_id),
                self.age_embedding(age_group),
                self.gender_embedding(gender),
                self.favorite_music_embedding(music_id),
                self.favorite_artist_embedding(artist_name),
            ],
            axis=1,
        )


class ItemModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.music_embedding = tf.keras.layers.Embedding(
            input_dim=num_music, output_dim=embedding_dim_music
        )
        self.genre_embedding = tf.keras.layers.Embedding(
            input_dim=num_genres, output_dim=embedding_dim_genre
        )
        self.audio_features = tf.keras.layers.Dense(embedding_dim_audio_features)

    def call(self, music_id, genre, audio_features):
        return tf.concat(
            [
                self.music_embedding(music_id),
                self.genre_embedding(genre),
                self.audio_features(audio_features),
            ],
            axis=1,
        )


# Define the full model
class RecommenderModel(tfrs.models.Model):
    def __init__(self, user_model, item_model):
        super().__init__()

        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=[
                tfrs.metrics.FactorizedTopK(candidates=item_dataset.batch(128)),
                # tfrs.metrics.Precision@(k=10),
                # tfrs.metrics.RecallAtK(k=10),
                # tfrs.metrics.NDCG(),
            ]
        )

    def compute_loss(self, features, training=False):
        label = features["label"]
        user_embeddings = self.user_model(
            features["user_id"],
            features["age_group"],
            features["gender"],
            features["music_id"],
            features["artist_name"],
        )
        item_embeddings = self.item_model(
            features["music_id"], features["genre"], features["audio_features"]
        )
        return self.task(user_embeddings, item_embeddings)


# Instantiate and compile the model
user_model = UserModel()
item_model = ItemModel()
model = RecommenderModel(user_model, item_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Train the model
model.fit(train_dataset.batch(128), epochs=10)
model.evaluate(test_dataset.batch(128))


