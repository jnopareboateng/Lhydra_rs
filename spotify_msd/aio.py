import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from annoy import AnnoyIndex
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from collections import defaultdict

class FeatureProcessor:
    def __init__(self, numerical_cols: List[str], categorical_cols: List[str]):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scalers = {}
        self.encoders = {}
        self.embedding_dims = {}

    def fit(self, df: pd.DataFrame):
        # Initialize scalers for numerical columns
        for col in self.numerical_cols:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
            self.scalers[col].partial_fit(df[[col]])

        # Initialize encoders for categorical columns
        for col in self.categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                # Add unknown category
                unique_values = np.append(df[col].astype(str).unique(), ["<unknown>"])
                self.encoders[col].fit(unique_values)
            else:
                # Update with new categories
                current_classes = set(self.encoders[col].classes_)
                new_classes = set(df[col].astype(str).unique())
                all_classes = sorted(current_classes.union(new_classes))
                self.encoders[col].classes_ = np.array(all_classes)

            # Calculate embedding dimensions
            n_unique = len(self.encoders[col].classes_)
            self.embedding_dims[col] = min(50, int(pow(n_unique, 0.25)))

        # Add frequency-based filtering
        min_frequency = 5  # minimum occurrences
        for col in self.categorical_cols:
            value_counts = df[col].value_counts()
            frequent_categories = value_counts[value_counts >= min_frequency].index
            df[col] = df[col].where(df[col].isin(frequent_categories), "<rare>")

    def transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        processed_features = {}

        # Transform numerical features
        for col in self.numerical_cols:
            processed_features[col] = torch.FloatTensor(
                self.scalers[col].transform(df[[col]])
            )

        # Transform categorical features
        for col in self.categorical_cols:
            try:
                transformed = self.encoders[col].transform(df[col].astype(str))
            except ValueError:
                # Handle unknown categories
                unknown_mask = ~df[col].astype(str).isin(self.encoders[col].classes_)
                values = df[col].astype(str).copy()
                values[unknown_mask] = "<unknown>"
                transformed = self.encoders[col].transform(values)

            # Use '_encoded' suffix in keys
            processed_features[f"{col}_encoded"] = torch.LongTensor(transformed)

        return processed_features

class MultiTaskMusicRecommender(nn.Module):
    def __init__(
        self,
        feature_processor,
        n_genres: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        pretrained_embeddings: Optional[Dict] = None,
    ):
        super().__init__()
        self.feature_processor = feature_processor

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for col in feature_processor.categorical_cols:
            n_unique = len(feature_processor.encoders[col].classes_)
            embed_dim = feature_processor.embedding_dims[col]

            # Load pretrained embeddings if available
            if pretrained_embeddings and col in pretrained_embeddings:
                self.embeddings[col] = nn.Embedding.from_pretrained(
                    pretrained_embeddings[col], freeze=False
                )
            else:
                self.embeddings[col] = nn.Embedding(n_unique, embed_dim)

        # Shared layers
        input_dim = len(feature_processor.numerical_cols) + sum(
            feature_processor.embedding_dims.values()
        )

        self.shared_network = self._create_mlp(input_dim, hidden_dims)

        # Task-specific heads
        last_hidden = hidden_dims[-1]
        self.playcount_head = nn.Linear(last_hidden, 1)
        self.genre_head = nn.Linear(last_hidden, n_genres)
        self.ranking_head = nn.Linear(last_hidden, 1)

        # ANNOY index for similar song retrieval
        self.annoy_index = None
        self.song_ids = None

    def _create_mlp(self, input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = dim

        return nn.Sequential(*layers)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Process embeddings for categorical features
        embedded_features = []

        # Handle numerical features
        numerical_features = []
        for col in self.feature_processor.numerical_cols:
            numerical_features.append(features[col])

        if numerical_features:
            numerical_features = torch.cat(numerical_features, dim=1)
            embedded_features.append(numerical_features)

        # Handle categorical features
        for col in self.feature_processor.categorical_cols:
            embedding_layer = self.embeddings[col]
            embedded = embedding_layer(features[f"{col}_encoded"])
            embedded_features.append(embedded)

        # Concatenate all features
        x = torch.cat(embedded_features, dim=1)

        # Pass through shared network
        x = self.shared_network(x)

        # Save embeddings before task-specific heads
        embeddings = x.clone()

        # Task-specific outputs
        return {
            "playcount": self.playcount_head(x),
            "genre": self.genre_head(x),
            "ranking": self.ranking_head(x),
            "embedding": embeddings,
        }

    def build_annoy_index(
        self, embeddings: np.ndarray, song_ids: List[str], n_trees: int = 10
    ):
        embedding_dim = embeddings.shape[1]
        self.annoy_index = AnnoyIndex(embedding_dim, "angular")
        self.song_ids = song_ids

        for i, embedding in tqdm(enumerate(embeddings), desc="Building ANNOY index"):
            self.annoy_index.add_item(i, embedding)

        self.annoy_index.build(n_trees)

def calculate_metrics(predictions, targets, k_values=[5, 10]):
    """Calculate ranking metrics."""
    metrics = {}

    def dcg_at_k(r, k):
        """Calculate DCG@K for a single sample
        Args:
            r: Relevance scores in rank order
            k: Number of results to consider
        Returns:
            DCG@K
        """
        r = np.asarray(r, dtype=float)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    def ndcg_at_k(r, k):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(r, k) / dcg_max

    for k in k_values:
        # Sort predictions to get top-k items
        top_k_items = np.argsort(predictions)[-k:]

        # NDCG@k
        ndcg = ndcg_at_k(targets[top_k_items], k)
        metrics[f"ndcg@{k}"] = ndcg

        # Precision@k
        precision = np.mean(targets[top_k_items] > 0)
        metrics[f"precision@{k}"] = precision

        # Recall@k - Add zero check
        total_positives = np.sum(targets > 0)
        if total_positives > 0:
            recall = np.sum(targets[top_k_items] > 0) / total_positives
        else:
            recall = 0.0
        metrics[f"recall@{k}"] = recall

    # MAP - Add zero check
    ap_sum = 0.0
    relevant_count = 0
    total_positives = np.sum(targets > 0)
    
    if total_positives > 0:
        sorted_indices = np.argsort(predictions)[::-1]
        for i, idx in enumerate(sorted_indices):
            if targets[idx] > 0:
                relevant_count += 1
                ap_sum += relevant_count / (i + 1)
        metrics["map"] = ap_sum / total_positives
    else:
        metrics["map"] = 0.0

    return metrics

def train_model(model, train_loader, val_loader, device="cpu"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    device = next(model.parameters()).device  # Get the device model is on

    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training Batches"):
        optimizer.zero_grad()
        # Move batch to the same device as model
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)
        loss = 0.0

        # Calculate ranking loss
        if "ranking" in outputs:
            ranking_loss = F.binary_cross_entropy_with_logits(
                outputs["ranking"], batch["ranking"]
            )
            loss += ranking_loss

        # Calculate playcount loss
        if "playcount" in outputs:
            playcount_loss = F.mse_loss(outputs["playcount"], batch["playcount"])
            loss += playcount_loss

        # Calculate genre loss
        if "genre" in outputs:
            genre_loss = F.cross_entropy(outputs["genre"], batch["genre"])
            loss += genre_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Training Loss: {total_loss / len(train_loader):.4f}")

class SongEmbeddingRetriever:
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.index = None
        self.song_ids = None

    def build_index(self, embeddings: np.ndarray, song_ids: List[str]):
        self.song_ids = song_ids
        embedding_dim = embeddings.shape[1]

        self.index = AnnoyIndex(embedding_dim, "angular")
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)

        self.index.build(self.n_trees)

    def get_similar_songs(
        self, embedding: np.ndarray, n_neighbors: int = 10
    ) -> Tuple[List[str], List[float]]:
        indices, distances = self.index.get_nns_by_vector(
            embedding, n_neighbors, include_distances=True
        )
        similar_songs = [self.song_ids[idx] for idx in indices]
        return similar_songs, distances

class MusicRecommenderSystem:
    def __init__(
        self,
        model: MultiTaskMusicRecommender,
        feature_processor: FeatureProcessor,
        song_retriever: SongEmbeddingRetriever,
    ):
        self.model = model
        self.feature_processor = feature_processor
        self.song_retriever = song_retriever

    def get_recommendations(
        self, user_data: pd.DataFrame, n_recommendations: int = 10
    ) -> pd.DataFrame:
        # Process features
        features = self.feature_processor.transform(user_data)
        features = {k: v.to(self.model.device) for k, v in features.items()}

        # Generate user embedding
        with torch.no_grad():
            outputs = self.model(features)
            user_embedding = outputs["embedding"].cpu().numpy().flatten()

        # Get similar songs
        similar_songs, distances = self.song_retriever.get_similar_songs(
            user_embedding, n_recommendations
        )

        # Create recommendations DataFrame
        recommendations = pd.DataFrame(
            {
                "track_id": similar_songs,
                "similarity_score": 1 - np.array(distances),
            }
        )

        return recommendations

class DataChunkLoader:

    def __init__(self, file_path: str, chunk_size: int):

        self.file_path = file_path

        self.chunk_size = chunk_size

        self._total_chunks = None

    @property
    def total_chunks(self):

        if self._total_chunks is None:

            # Calculate total number of chunks based on file size

            total_rows = sum(1 for _ in open(self.file_path)) - 1  # -1 for header

            self._total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size

        return self._total_chunks

    def load_chunks(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield chunk

def evaluate_model(model, test_loader, metrics=['ndcg', 'map', 'precision', 'recall']):
    results = defaultdict(list)
    model.eval()
    device = next(model.parameters()).device  # Get the device model is on
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to the same device as model
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            
            # Move predictions and targets back to CPU for metric calculation
            metric_values = calculate_metrics(
                outputs['ranking'].cpu().numpy(),
                batch['ranking'].cpu().numpy()
            )
            for metric, value in metric_values.items():
                results[metric].append(value)
    
    return {k: np.mean(v) for k, v in results.items()}
