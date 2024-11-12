import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedListNetLoss(nn.Module):
    def __init__(self, k=10, ils_weight=0.1, temperature=1.0):
        super().__init__()
        self.k = k
        self.ils_weight = ils_weight
        self.temperature = temperature

    def forward(self, predictions, targets):
        # Extract plays predictions from dict if needed
        y_pred = predictions["plays"] if isinstance(predictions, dict) else predictions
        y_true = targets["plays"] if isinstance(targets, dict) else targets

        # Ensure predictions and targets are 2D
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        # Apply softmax along the correct dimension (batch dimension)
        P_y_pred = F.softmax(y_pred, dim=1)
        P_y_true = F.softmax(y_true, dim=1)

        listnet_loss = -torch.sum(P_y_true * torch.log(P_y_pred + 1e-10)) / y_pred.size(
            0
        )

        # For now, skip ILS calculation since we don't have features
        total_loss = listnet_loss
        ils_penalty = torch.tensor(0.0, device=y_pred.device)

        return total_loss, {
            "listnet_loss": listnet_loss.item(),
            "ils_penalty": ils_penalty.item(),
            "total_loss": total_loss.item(),
        }


class HybridRecommender(nn.Module):
    def __init__(
        self,
        num_genders,
        num_music_items,
        num_genres,
        num_artist_features,
        num_numerical_features,
        num_release_years,
        user_id_hashed_dim,
        embedding_dim=64,
        hidden_dims=None,
        dropout_prob=0.3,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        super(HybridRecommender, self).__init__()

        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.release_year_embedding = nn.Embedding(num_release_years, embedding_dim)

        self.artist_fc = nn.Linear(num_artist_features, embedding_dim)
        self.genre_fc = nn.Linear(num_genres, embedding_dim)
        self.music_fc = nn.Linear(num_music_items, embedding_dim)

        total_embedding_dim = embedding_dim * 5
        input_dim = total_embedding_dim + num_numerical_features + user_id_hashed_dim

        print(f"Input dimension calculated: {input_dim}")

        self.shared_layers = nn.ModuleList()
        current_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            self.shared_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )
            current_dim = hidden_dim

        self.plays_head = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(
        self,
        user_id_hashed,
        artist_features,
        gender_ids,
        music_features,
        genre_features,
        numerical_features,
        release_years,
    ):
        gender_embedded = self.gender_embedding(gender_ids)
        release_year_embedded = self.release_year_embedding(release_years)

        artist_embedded = self.artist_fc(artist_features)
        genre_embedded = self.genre_fc(genre_features)
        music_embedded = self.music_fc(music_features)

        combined = torch.cat(
            [
                user_id_hashed,
                gender_embedded,
                release_year_embedded,
                artist_embedded,
                genre_embedded,
                music_embedded,
                numerical_features,
            ],
            dim=1,
        )

        x = combined
        for layer in self.shared_layers:
            x = layer(x)

        plays_pred = self.plays_head(x)

        return {"plays": plays_pred}


class MultitaskLoss(nn.Module):
    def __init__(self):
        super(MultitaskLoss, self).__init__()
        self.criterion = EnhancedListNetLoss()

    def forward(self, predictions, targets):
        total_loss, losses_dict = self.criterion(predictions["plays"], targets["plays"])
        return total_loss, losses_dict
