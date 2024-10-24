import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import DataPreprocessor
from tensorflow_docs.model import HybridRecommender, ListNetLoss
import pickle

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def main():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the preprocessor
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    
    # Load and preprocess data
    filepath = "../data/cleaned_modv2.csv"
    data = preprocessor.load_data(filepath)
    data_encoded = preprocessor.encode_features(data)
    features = preprocessor.feature_engineering(data_encoded)
    train_features, val_features, test_features, train_target, val_target, test_target = preprocessor.split_data(features)

    # Save preprocessors
    preprocessor.save_preprocessors(directory="models/")

    # Convert data to PyTorch tensors and move to device
    train_user_ids = torch.tensor(train_features['user_id_encoded'].values, dtype=torch.long).to(device)
    train_item_ids = torch.tensor(train_features['track_encoded'].values, dtype=torch.long).to(device)
    train_genre_ids = torch.tensor(train_features['genre_encoded'].values, dtype=torch.long).to(device)
    train_artist_features = torch.tensor(train_features[[col for col in train_features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device)
    train_numerical_features = torch.tensor(train_features[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                                            'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                                                            'time_signature', 'explicit']].values, dtype=torch.float).to(device)
    train_target = torch.tensor(train_target.values, dtype=torch.float).unsqueeze(1).to(device)

    val_user_ids = torch.tensor(val_features['user_id_encoded'].values, dtype=torch.long).to(device)
    val_item_ids = torch.tensor(val_features['track_encoded'].values, dtype=torch.long).to(device)
    val_genre_ids = torch.tensor(val_features['genre_encoded'].values, dtype=torch.long).to(device)
    val_artist_features = torch.tensor(val_features[[col for col in val_features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device)
    val_numerical_features = torch.tensor(val_features[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                                        'mode','speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                     j                                   'time_signature', 'explicit']].values, dtype=torch.float).to(device)
    val_target = torch.tensor(val_target.values, dtype=torch.float).unsqueeze(1).to(device)

    test_user_ids = torch.tensor(test_features['user_id_encoded'].values, dtype=torch.long).to(device)
    test_item_ids = torch.tensor(test_features['track_encoded'].values, dtype=torch.long).to(device)
    test_genre_ids = torch.tensor(test_features['genre_encoded'].values, dtype=torch.long).to(device)
    test_artist_features = torch.tensor(test_features[[col for col in test_features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device)
    test_numerical_features = torch.tensor(test_features[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                                          'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                                                          'time_signature', 'explicit']].values, dtype=torch.float).to(device)
    test_target = torch.tensor(test_target.values, dtype=torch.float).unsqueeze(1).to(device)

    print(f"Max user_id: {train_user_ids.max().item()}")
    print(f"Max artist_id: {train_artist_features.max().item()}")
    print(f"Max item_id: {train_item_ids.max().item()}")
    print(f"Max genre_id: {train_genre_ids.max().item()}")

    # Create validation tensors and DataLoader
    val_dataset = TensorDataset(
        val_user_ids,
        val_item_ids,
        val_genre_ids,
        val_artist_features,
        val_numerical_features,
        val_target,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize the model
    num_users = len(preprocessor.user_id_encoder.classes_)
    num_genres = len(preprocessor.genre_encoder.classes_)
    num_tracks = len(preprocessor.track_encoder.classes_)
    num_artist_features = train_artist_features.shape[1]
    num_numerical_features = train_numerical_features.shape[1]

    embedding_dim = 32
    num_layers = 3
    hidden_dims = [256, 128, 64]
    dropout_prob = 0.2

    model = HybridRecommender(
        num_users,
        num_artist_features,
        num_tracks,
        num_genres,
        num_numerical_features,
        embedding_dim,
        num_layers,
        hidden_dims,
        dropout_prob
    ).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Define loss function
    criterion = ListNetLoss(k=10)

    # Create DataLoader for batch processing
    dataset = TensorDataset(
        train_user_ids,
        train_item_ids,
        train_genre_ids,
        train_artist_features,
        train_numerical_features,
        train_target,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    val_losses = []  # Use this for storing validation losses

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (
            batch_user_ids,
            batch_item_ids,
            batch_genre_ids,
            batch_artist_features,
            batch_numerical_features,
            batch_target,
        ) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(
                batch_user_ids,
                batch_artist_features,
                batch_item_ids,
                batch_genre_ids,
                batch_numerical_features,
            )
            
            loss = criterion(predictions, batch_target)

            # Add L2 regularization
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_user_ids, batch_item_ids, batch_genre_ids, batch_artist_features, batch_numerical_features, batch_targets = batch
                predictions = model(
                    batch_user_ids, batch_artist_features, batch_item_ids, batch_genre_ids, batch_numerical_features
                )
                batch_loss = criterion(predictions, batch_targets)
                val_loss += batch_loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(dataloader):.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(
            test_user_ids,
            test_artist_features,
            test_item_ids,
            test_genre_ids,
            test_numerical_features,
        )
        ndcg_score = ndcg_at_k(
            test_target.cpu().numpy(), predictions.cpu().numpy(), k=10
        )
        print(f"Test NDCG@10: {ndcg_score:.4f}")
        precision_score = precision_at_k(
            test_target.cpu().numpy(), predictions.cpu().numpy(), k=10
        )
        print(f"Test Precision@10: {precision_score:.4f}")
        recall_score = recall_at_k(
            test_target.cpu().numpy(), predictions.cpu().numpy(), k=10
        )
        print(f"Test Recall@10: {recall_score:.4f}")
        f1_score = f1_score_at_k(
            test_target.cpu().numpy(), predictions.cpu().numpy(), k=10
        )
        print(f"Test F1-score@10: {f1_score:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved to 'models/model.pth'")
    print("Training complete!")

def ndcg_at_k(y_true, y_pred, k):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        y_true (np.array): Ground truth scores.
        y_pred (np.array): Predicted scores.
        k (int): Rank cutoff.

    Returns:
        float: NDCG score.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[indices]

    gains = np.log2(y_true_sorted + 1)
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    dcg = np.sum(gains[:k] / discounts[:k])

    ideal_gains = np.log2(np.sort(y_true)[::-1] + 1)
    ideal_dcg = np.sum(ideal_gains[:k] / discounts[:k])

    return dcg / ideal_dcg if ideal_dcg != 0 else 0.0


def precision_at_k(y_true, y_pred, k):
    """
    Precision@k implementation.

    Parameters:
    - y_true (np.array): Ground truth scores.
    - y_pred (np.array): Predicted scores.
    - k (int): Rank cutoff.

    Returns:
    - float: Recall score.
    """
    # Precision@10 implementation
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    indices = np.argsort(y_pred)[::-1]
    return np.sum(y_true[indices[:k]]) / k


def recall_at_k(y_true, y_pred, k):
    """
    Recall@k implementation.

    Parameters:
    - y_true (np.array): Ground truth scores.
    - y_pred (np.array): Predicted scores.
    - k (int): Rank cutoff.

    Returns:
    - float: Recall score.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    indices = np.argsort(y_pred)[::-1]  # Define indices here
    return np.sum(y_true[indices[:k]]) / len(y_true)


def f1_score_at_k(y_true, y_pred, k):
    """
    F1@k implementation.

    Parameters:
    - y_true (np.array): Ground truth scores.
    - y_pred (np.array): Predicted scores.
    - k (int): Rank cutoff.

    Returns:
    - float: Recall score.
    """
    # F1-score@10 implementation
    return (precision_at_k(y_true, y_pred, k) + recall_at_k(y_true, y_pred, k)) / 2


def evaluate_model(predictions, test_target):
    """
    Evaluate the model using NDCG@10, Precision@10, Recall@10, and F1-score@10.

    Args:
        predictions (np.array): Predicted scores.
        test_target (np.array): Ground truth scores.
    """
    ndcg_score = ndcg_at_k(test_target, predictions, k=10)
    precision_score = precision_at_k(test_target, predictions, k=10)
    recall_score = recall_at_k(test_target, predictions, k=10)
    f1_score = f1_score_at_k(test_target, predictions, k=10)
    print(f"NDCG@10: {ndcg_score:.4f}")
    print(f"Precision@10: {precision_score:.4f}")
    print(f"Recall@10: {recall_score:.4f}")
    print(f"F1-score@10: {f1_score:.4f}")


if __name__ == "__main__":
    main()
