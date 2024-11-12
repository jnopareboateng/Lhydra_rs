import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNetLoss(nn.Module):
    def __init__(self, k=10):
        super(ListNetLoss, self).__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)


class HybridRecommender(nn.Module):
    def __init__(
        self,
        num_genders,
        num_music_items,  # This should match len(music_feature_cols)
        num_genres,
        num_artist_features,
        num_numerical_features,
        num_release_years,
        embedding_dim,
        num_layers=4,
        hidden_dims=None,
        dropout_prob=0.3,
        use_batch_norm=True,
    ):
        super(HybridRecommender, self).__init__()
        
        # Print dimensions for debugging
        print(f"Initializing model with music features: {num_music_items}")
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        # Embeddings
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.release_year_embedding = nn.Embedding(num_release_years, embedding_dim)

        # Feature transformations
        self.music_fc = nn.Linear(num_music_items, embedding_dim)
        self.genre_fc = nn.Linear(num_genres, embedding_dim)
        self.artist_fc = nn.Linear(num_artist_features, embedding_dim)

        # Calculate expected dimension
        self.expected_dim = (
            32  # user_id_hashed
            + embedding_dim  # gender
            + embedding_dim  # release_year
            + embedding_dim  # music
            + embedding_dim  # artist
            + embedding_dim  # genre
            + num_numerical_features  # numerical
        )

        self.input_bn = (
            nn.BatchNorm1d(self.expected_dim) if use_batch_norm else nn.Identity()
        )

        # Create layers
        layers = []
        input_dim = self.expected_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dims[i]

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)

    def forward(
        self,
        user_id_hashed,
        artist_features,
        gender_id,
        music_features,
        genre_features,
        numerical_features,
        release_year,
    ):
        # Embed features
        gender_embedded = self.gender_embedding(gender_id.squeeze(-1))
        # gender_embedded = self.gender_embedding(gender_id.squeeze(-1).long())  # Convert to Long type
        release_year_embedded = self.release_year_embedding(release_year.squeeze(-1))
        music_embedded = self.music_fc(music_features)
        artist_embedded = self.artist_fc(artist_features)
        genre_embedded = self.genre_fc(genre_features)

        # Concatenate all features
        concat_features = torch.cat(
            [
                user_id_hashed,
                gender_embedded,
                release_year_embedded,
                music_embedded,
                artist_embedded,
                genre_embedded,
                numerical_features,
            ],
            dim=1,
        )

        # Verify dimensions
        assert (
            concat_features.shape[1] == self.expected_dim
        ), f"Expected {self.expected_dim} features but got {concat_features.shape[1]}"

        # Process through network
        x = self.input_bn(concat_features)
        return self.fc(x)
