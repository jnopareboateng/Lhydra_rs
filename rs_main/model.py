import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNetLoss(nn.Module):
    def __init__(self, k=10):
        super(ListNetLoss, self).__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        # Remove the softmax operations as they're causing numerical instability
        # and might not be appropriate for this regression task
        return F.mse_loss(y_pred, y_true)
        # return F.rms_norm(y_pred, y_true)


class HybridRecommender(nn.Module):
    def __init__(
        self,
        num_users,
        num_artist_features,
        num_genres,
        num_numerical_features,
        embedding_dim,
        num_layers=4,
        hidden_dims=None,
        dropout_prob=0.3,
        use_batch_norm=True,
    ):
        super(HybridRecommender, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.artist_fc = nn.Linear(num_artist_features, embedding_dim)
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(embedding_dim * 3 + num_numerical_features)
        
        # Create layers with batch normalization
        layers = []
        input_dim = embedding_dim * 3 + num_numerical_features
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dims[i]
        
        # Final layer with sigmoid activation for normalized output
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.ReLU())
        
        self.fc = nn.Sequential(*layers)

    def forward(self, user_id, artist_features, genre_id, numerical_features):
        # Ensure all inputs are 2-dimensional
        user_id = user_id.view(-1, 1)
        genre_id = genre_id.view(-1, 1)
        
        if numerical_features.dim() == 1:
            numerical_features = numerical_features.unsqueeze(0)
        
        user_embedded = self.user_embedding(user_id).squeeze(1)
        artist_embedded = self.artist_fc(artist_features)
        genre_embedded = self.genre_embedding(genre_id).squeeze(1)

        # Concatenate all features
        concat_features = torch.cat(
            (user_embedded, artist_embedded, genre_embedded, numerical_features),
            dim=1
        )
        
        # Apply input batch normalization
        concat_features = self.input_bn(concat_features)
        
        # Forward through the network
        output = self.fc(concat_features)
        return output
