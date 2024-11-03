from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess categorical features
le = LabelEncoder()
train_df['gender'] = le.fit_transform(train_df['gender'])
val_df['gender'] = le.transform(val_df['gender'])

train_df['genre'] = le.fit_transform(train_df['genre'])
val_df['genre'] = le.transform(val_df['genre'])

# Handle artist_name (choose one approach)
# Approach 1: Pre-trained embedding (if available)
# artist_embedding =...  # Load pre-trained embedding

# Approach 2: Categorical encoding
train_df['artist_name'] = le.fit_transform(train_df['artist_name'])
val_df['artist_name'] = le.transform(val_df['artist_name'])

# Embedding Tower
class MusicEmbeddingTower(nn.Module):
    def __init__(self, num_users, num_genders, num_genres, num_artists, audio_features_dim):
        super(MusicEmbeddingTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 64)  # Adjust embedding dim as needed
        self.gender_embedding = nn.Embedding(num_genders, 16)
        self.genre_embedding = nn.Embedding(num_genres, 32)
        self.artist_embedding = nn.Embedding(num_artists, 64)  # Adjust embedding dim as needed
        self.audio_features_fc = nn.Linear(audio_features_dim, 128)  # Adjust output dim as needed

    def forward(self, user_ids, genders, genres, artist_ids, audio_features):
        user_embeddings = self.user_embedding(user_ids)
        gender_embeddings = self.gender_embedding(genders)
        genre_embeddings = self.genre_embedding(genres)
        artist_embeddings = self.artist_embedding(artist_ids)
        audio_features_embeddings = torch.relu(self.audio_features_fc(audio_features))
        return {
            'user': user_embeddings,
            'gender': gender_embeddings,
            'genre': genre_embeddings,
            'artist': artist_embeddings,
            'audio_features': audio_features_embeddings
        }

# Prediction Tower (Hybrid Approach)
# class MusicPredictionTower(nn.Module):
#     def __init__(self, embedding_dim, output_dim):
#         super(MusicPredictionTower, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, 256)  # Adjust input/output dim as needed
#         self.fc2 = nn.Linear(256, output_dim)

#     def forward(self, user_demographics, music_item_embeddings):
#         x = torch.cat((user_demographics, music_item_embeddings), dim=1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    # Updated MusicEmbeddingTower
class MusicEmbeddingTower(nn.Module):
    def __init__(self, num_users, num_genders, num_genres, num_artists, audio_features_dim):
        super(MusicEmbeddingTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 64)
        self.gender_embedding = nn.Embedding(num_genders, 16)
        self.genre_embedding = nn.Embedding(num_genres, 32)
        self.artist_embedding = nn.Embedding(num_artists, 64)
        self.audio_features_fc = nn.Sequential(
            nn.Linear(audio_features_dim, 256),  # Input dim: 14 (audio features)
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, user_ids, genders, genres, artist_ids, audio_features):
        user_embeddings = self.user_embedding(user_ids)
        gender_embeddings = self.gender_embedding(genders)
        genre_embeddings = self.genre_embedding(genres)
        artist_embeddings = self.artist_embedding(artist_ids)
        audio_features_embeddings = self.audio_features_fc(audio_features)
        return {
            'user': user_embeddings,
            'gender': gender_embeddings,
            'genre': genre_embeddings,
            'artist': artist_embeddings,
            'audio_features': audio_features_embeddings
        }

# Updated MusicRecommenderModel (outputting ranking scores)
class MusicRecommenderModel(nn.Module):
    def __init__(self, num_users, num_genders, num_genres, num_artists, audio_features_dim):
        super(MusicRecommenderModel, self).__init__()
        self.embedding_tower = MusicEmbeddingTower(num_users, num_genders, num_genres, num_artists, audio_features_dim)
        self.ranking_fc = nn.Linear(128 + 64 + 16 + 32 + 64, 1)  # Input dim: sum of embedding dims

    def forward(self, user_ids, genders, genres, artist_ids, audio_features):
        embeddings = self.embedding_tower(user_ids, genders, genres, artist_ids, audio_features)
        x = torch.cat((embeddings['user'], embeddings['gender'], embeddings['genre'], embeddings['artist'], embeddings['audio_features']), dim=1)
        ranking_scores = self.ranking_fc(x)
        return ranking_scores
    
# Create Model
class MusicRecommenderModel(nn.Module):
    def __init__(self, num_users, num_genders, num_genres, num_artists, audio_features_dim, output_dim):
        super(MusicRecommenderModel, self).__init__()
        self.embedding_tower = MusicEmbeddingTower(num_users, num_genders, num_genres, num_artists, audio_features_dim)
        self.prediction_tower = MusicPredictionTower(256, output_dim)  # Adjust input dim as needed

    def forward(self, user_ids, genders, genres, artist_ids, audio_features):
        embeddings = self.embedding_tower(user_ids, genders, genres, artist_ids, audio_features)
        user_demographics = torch.cat((embeddings['user'], embeddings['gender']), dim=1)
        music_item_embeddings = torch.cat((embeddings['genre'], embeddings['artist'], embeddings['audio_features']), dim=1)
        return self.prediction_tower(user_demographics, music_item_embeddings)

# Define Loss Function and Optimizer
model = MusicRecommenderModel(num_users=train_df['user_id'].nunique(),
                              num_genders=train_df['gender'].nunique(),
                              num_genres=train_df['genre'].nunique(),
                              num_artists=train_df['artist_name'].nunique(),
                              audio_features_dim=len(audio_features_columns),
                              output_dim=1)  # Adjust output dim as needed (e.g., number of music items)

criterion = nn.MSELoss()  # Choose a suitable loss function (e.g., BCEWithLogitsLoss for binary classification)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
for epoch in range(10):  # Adjust number of epochs as needed
    optimizer.zero_grad()
    outputs = model(train_df['user_id'], train_df['gender'], train_df['genre'], train_df['artist_name'], train_df[audio_features_columns])
    loss = criterion(outputs, train_df['target'])  # Assuming a target variable (e.g., rating, binary label)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

import torch
from sklearn.metrics import precision_score, recall_score
from torchrec.metrics import ndcg

# Assume 'odel' is the trained MusicRecommenderModel
# Assume 'test_data' is a PyTorch DataLoader with the test dataset

# Evaluate Precision@K, Recall@K, and NDCG
k_values = [5, 10, 20]
for k in k_values:
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []
    with torch.no_grad():
        for batch in test_data:
            user_ids, genders, genres, artist_ids, audio_features, labels = batch
            ranking_scores = model(user_ids, genders, genres, artist_ids, audio_features)
            _, top_k_indices = torch.topk(ranking_scores, k, dim=1)
            top_k_labels = labels.gather(1, top_k_indices)
            precision_at_k.append(precision_score(top_k_labels, torch.ones_like(top_k_labels), zero_division=0))
            recall_at_k.append(recall_score(top_k_labels, torch.ones_like(top_k_labels), zero_division=0))
            ndcg_at_k.append(ndcg(ranking_scores, labels, k))
    print(f"P@{k}: {torch.mean(torch.tensor(precision_at_k)):.4f}")
    print(f"R@{k}: {torch.mean(torch.tensor(recall_at_k)):.4f}")
    print(f"NDCG@{k}: {torch.mean(torch.tensor(ndcg_at_k)):.4f}")

# Evaluate MAP, MRR, and HR (additional metrics)
#... (implementation similar to above)