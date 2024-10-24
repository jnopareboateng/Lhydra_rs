import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNetLoss(nn.Module):
    def __init__(self, k=10):
        super(ListNetLoss, self).__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.softmax(y_true, dim=1)
        return -torch.sum(y_true * torch.log(y_pred + 1e-10), dim=1).mean()


class HybridRecommender(nn.Module):
    def __init__(
        self,
        num_users,
        num_artist_features,
        num_tracks,
        num_genres,
        num_numerical_features,
        embedding_dim,
        num_layers=3,
        hidden_dims=None,
        dropout_prob=0.2,
    ):
        super(HybridRecommender, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.bn = nn.BatchNorm1d(num_features=143)  # Apply Batch Normalization
        
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.artist_fc = nn.Linear(num_artist_features, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 4 + num_numerical_features
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dims[i]
        layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, user_id, artist_features, track_id, genre_id, numerical_features):
        # Ensure all inputs are 2-dimensional
        user_id = user_id.view(-1, 1)
        track_id = track_id.view(-1, 1)
        genre_id = genre_id.view(-1, 1)
        
        if numerical_features.dim() == 1:
            numerical_features = numerical_features.unsqueeze(0)
        
        user_embedded = self.user_embedding(user_id).squeeze(1)
        artist_embedded = self.artist_fc(artist_features)
        track_embedded = self.track_embedding(track_id).squeeze(1)
        genre_embedded = self.genre_embedding(genre_id).squeeze(1)

        concat_features = torch.cat(
            (user_embedded, artist_embedded, track_embedded, genre_embedded, numerical_features),
            dim=1
        )
        normalized_features = self.bn(concat_features)  # Normalize features

        output = self.fc(normalized_features)
        # return output.squeeze()
        return output
