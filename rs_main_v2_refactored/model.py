import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedListNetLoss(nn.Module):
    def __init__(self, k=10, ils_weight=0.1, temperature=1.0):
        """
        Enhanced ListNet Loss with Intra-List Similarity regularization

        Args:
            k (int): Top-k items to consider
            ils_weight (float): Weight for the ILS regularization term
            temperature (float): Temperature for similarity scaling
        """
        super(EnhancedListNetLoss, self).__init__()
        self.k = k
        self.ils_weight = ils_weight
        self.temperature = temperature

    def compute_similarity_matrix(self, features):
        """
        Compute pairwise similarities between items in the batch

        Args:
            features (torch.Tensor): Combined feature representation [batch_size, feature_dim]
        """
        # Normalize features
        normalized_features = F.normalize(features, p=2, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())

        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature

        return similarity_matrix

    def compute_ils_penalty(self, similarity_matrix, rankings):
        """
        Compute ILS penalty based on item similarities and their positions in ranking

        Args:
            similarity_matrix (torch.Tensor): Pairwise similarity matrix [batch_size, batch_size]
            rankings (torch.Tensor): Predicted rankings [batch_size, 1]
        """
        batch_size = rankings.size(0)

        # Convert rankings to pairwise position differences
        position_diff = (rankings - rankings.t()).abs()

        # Weight similarities by position differences (closer positions = higher penalty)
        position_weights = torch.exp(-position_diff)

        # Compute penalty: high similarity items should be far apart in ranking
        ils_penalty = (similarity_matrix * position_weights).sum() / (
            batch_size * (batch_size - 1)
        )

        return ils_penalty

    def combine_features(self, genre_features, artist_features, music_features):
        """
        Combine different feature types with appropriate weighting
        """
        # Normalize each feature type
        genre_norm = F.normalize(genre_features, p=2, dim=1)
        artist_norm = F.normalize(artist_features, p=2, dim=1)
        music_norm = F.normalize(music_features, p=2, dim=1)

        # Combine features with weights
        # You can adjust these weights based on importance
        combined = torch.cat(
            [
                genre_norm * 0.4,  # Higher weight for genre diversity
                artist_norm * 0.3,
                music_norm * 0.3,
            ],
            dim=1,
        )

        return combined

    def forward(self, y_pred, y_true, genre_features, artist_features, music_features):
        """
        Forward pass computing both ListNet loss and ILS regularization

        Args:
            y_pred (torch.Tensor): Predicted scores [batch_size, 1]
            y_true (torch.Tensor): True scores [batch_size, 1]
            genre_features (torch.Tensor): Genre TF-IDF features
            artist_features (torch.Tensor): Artist TF-IDF features
            music_features (torch.Tensor): Music TF-IDF features
        """
        # Original ListNet loss
        P_y_pred = F.softmax(y_pred, dim=0)
        P_y_true = F.softmax(y_true, dim=0)
        listnet_loss = -torch.sum(P_y_true * torch.log(P_y_pred + 1e-10)) / y_pred.size(
            0
        )

        # Compute ILS penalty
        combined_features = self.combine_features(
            genre_features, artist_features, music_features
        )
        similarity_matrix = self.compute_similarity_matrix(combined_features)
        ils_penalty = self.compute_ils_penalty(similarity_matrix, y_pred)

        # Combine losses
        total_loss = listnet_loss + self.ils_weight * ils_penalty

        return total_loss, {
            "listnet_loss": listnet_loss.item(),
            "ils_penalty": ils_penalty.item(),
            "total_loss": total_loss.item(),
        }

    def get_diversity_metric(
        self, genre_features, artist_features, music_features, y_pred, k=10
    ):
        """
        Compute diversity metric for top-k recommendations
        """
        combined_features = self.combine_features(
            genre_features, artist_features, music_features
        )
        similarity_matrix = self.compute_similarity_matrix(combined_features)

        # Get top-k indices
        _, top_k_indices = torch.topk(y_pred, k=min(k, y_pred.size(0)))

        # Compute average similarity between top-k items (lower is more diverse)
        top_k_similarities = similarity_matrix[top_k_indices][:, top_k_indices]
        diversity_score = 1.0 - (top_k_similarities.sum() - k) / (k * (k - 1))

        return diversity_score.item()


class HybridRecommender(nn.Module):
    def __init__(
        self,
        num_genders,
        num_music_items,
        num_genres,
        num_artist_features,
        num_numerical_features,
        num_release_years,
        embedding_dim,
        hidden_dims=[512, 256, 128, 64],
        dropout_prob=0.3,
    ):
        super(HybridRecommender, self).__init__()

        # Embeddings
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.release_year_embedding = nn.Embedding(num_release_years, embedding_dim)

        # Feature transformations with batch norm
        self.music_fc = nn.Sequential(
            nn.Linear(num_music_items, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.genre_fc = nn.Sequential(
            nn.Linear(num_genres, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.artist_fc = nn.Sequential(
            nn.Linear(num_artist_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

        # Calculate expected dimension
        self.expected_dim = 32 + embedding_dim * 5 + num_numerical_features

        # Input normalization
        self.input_bn = nn.BatchNorm1d(self.expected_dim)

        # Main network layers with residual connections
        self.layer1 = nn.Sequential(
            nn.Linear(self.expected_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.8),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.6),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.4),
        )

        # Residual connections
        self.res1 = nn.Linear(self.expected_dim, hidden_dims[1])
        self.res2 = nn.Linear(hidden_dims[0], hidden_dims[2])
        self.res3 = nn.Linear(hidden_dims[1], hidden_dims[3])

        # Output layer
        self.output = nn.Sequential(nn.Linear(hidden_dims[3], 1), nn.Sigmoid())

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
        # Ensure all inputs are 2D
        if user_id_hashed.dim() == 1:
            user_id_hashed = user_id_hashed.unsqueeze(1)
        if gender_id.dim() == 1:
            gender_id = gender_id.unsqueeze(1)
        if release_year.dim() == 1:
            release_year = release_year.unsqueeze(1)
        if numerical_features.dim() == 1:
            numerical_features = numerical_features.unsqueeze(1)

        # Process embeddings
        gender_embedded = self.gender_embedding(gender_id.long().squeeze(-1))
        release_year_embedded = self.release_year_embedding(release_year.long().squeeze(-1))
        
        # Process features through FC layers
        music_embedded = self.music_fc(music_features.float())
        artist_embedded = self.artist_fc(artist_features.float())
        genre_embedded = self.genre_fc(genre_features.float())

        # Concatenate all features
        concat_features = torch.cat(
            [
                user_id_hashed.float(),
                gender_embedded,
                release_year_embedded,
                music_embedded,
                artist_embedded,
                genre_embedded,
                numerical_features.float(),
            ],
            dim=1
        )

        # Add dimension checks
        expected_batch_size = user_id_hashed.size(0)
        assert all(x.size(0) == expected_batch_size for x in [
            gender_embedded, release_year_embedded, music_embedded,
            artist_embedded, genre_embedded, numerical_features
        ]), "Batch size mismatch in features"
        
        assert concat_features.shape[1] == self.expected_dim, \
            f"Expected {self.expected_dim} features but got {concat_features.shape[1]}"

        # Rest of the forward pass remains the same
        x = self.input_bn(concat_features)
        x1 = self.layer1(x)
        r1 = self.res1(x)
        x2 = self.layer2(x1)
        x2 = x2 + r1
        x3 = self.layer3(x2)
        r2 = self.res2(x1)
        x3 = x3 + r2
        x4 = self.layer4(x3)
        r3 = self.res3(x2)
        x4 = x4 + r3
        
        return self.output(x4)
