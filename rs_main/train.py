import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import DataPreprocessor
from model import HybridRecommender, ListNetLoss
import pickle

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path='models/best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')  # Initialize with infinity
        self.early_stop = False
        self.save_path = save_path
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:  # Changed condition
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)

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
    train_genre_ids = torch.tensor(train_features['genre_encoded'].values, dtype=torch.long).to(device)
    train_artist_features = torch.tensor(train_features[[col for col in train_features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device)
    train_numerical_features = torch.tensor(train_features[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                                            'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                                                            'time_signature', 'explicit']].values, dtype=torch.float).to(device)
    train_target = torch.tensor(train_target.values, dtype=torch.float).unsqueeze(1).to(device)

    val_user_ids = torch.tensor(val_features['user_id_encoded'].values, dtype=torch.long).to(device)
    val_genre_ids = torch.tensor(val_features['genre_encoded'].values, dtype=torch.long).to(device)
    val_artist_features = torch.tensor(val_features[[col for col in val_features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device)
    val_numerical_features = torch.tensor(val_features[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                                        'mode','speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                                                        'time_signature', 'explicit']].values, dtype=torch.float).to(device)
    val_target = torch.tensor(val_target.values, dtype=torch.float).unsqueeze(1).to(device)

    test_user_ids = torch.tensor(test_features['user_id_encoded'].values, dtype=torch.long).to(device)
    test_genre_ids = torch.tensor(test_features['genre_encoded'].values, dtype=torch.long).to(device)
    test_artist_features = torch.tensor(test_features[[col for col in test_features.columns if col.startswith('artist_tfidf_')]].values, dtype=torch.float).to(device)
    test_numerical_features = torch.tensor(test_features[['age', 'duration', 'acousticness', 'danceability', 'energy', 'key', 'loudness', 
                                                          'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                                                          'time_signature', 'explicit']].values, dtype=torch.float).to(device)
    test_target = torch.tensor(test_target.values, dtype=torch.float).unsqueeze(1).to(device)

    print(f"Max user_id: {train_user_ids.max().item()}")
    print(f"Max artist_id: {train_artist_features.max().item()}")
    print(f"Max genre_id: {train_genre_ids.max().item()}")

    # Create validation tensors and DataLoader
    val_dataset = TensorDataset(
        val_user_ids,
        val_genre_ids,
        val_artist_features,
        val_numerical_features,
        val_target,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize the model
    num_users = len(preprocessor.user_id_encoder.classes_)
    num_genres = len(preprocessor.genre_encoder.classes_)
    num_artist_features = train_artist_features.shape[1]
    num_numerical_features = train_numerical_features.shape[1]

    embedding_dim = 256  # Increased embedding dimension
    num_layers = 4
    hidden_dims = [512, 256, 128, 32]  # Larger hidden dimensions
    dropout_prob = 0.3  # Increased dropout

    # Initialize the model with batch normalization
    model = HybridRecommender(
        num_users,
        num_artist_features,
        num_genres,
        num_numerical_features,
        embedding_dim,
        num_layers,
        hidden_dims,
        dropout_prob,
        use_batch_norm=True  # Enable batch normalization
    ).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6
    )

    # Define loss function
    criterion = ListNetLoss(k=10)

    # Create DataLoader for batch processing
    dataset = TensorDataset(
        train_user_ids,
        train_genre_ids,
        train_artist_features,
        train_numerical_features,
        train_target,
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path='models/best_model.pth')
    val_losses = []

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training phase
        for batch_idx, (batch_user_ids, batch_genre_ids, batch_artist_features, 
                       batch_numerical_features, batch_target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            predictions = model(
                batch_user_ids,
                batch_artist_features,
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
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_user_ids, batch_genre_ids, batch_artist_features, batch_numerical_features, batch_targets = batch
                predictions = model(
                    batch_user_ids, batch_artist_features, batch_genre_ids, batch_numerical_features
                )
                batch_loss = criterion(predictions, batch_targets)
                val_loss += batch_loss.item()
        
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        # Early Stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    # Load the best model before evaluation
    model.load_state_dict(torch.load('models/best_model.pth'))

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(
            test_user_ids,
            test_artist_features,
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
    """Modified NDCG@k implementation."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Get predicted ranking
    pred_indices = np.argsort(y_pred)[::-1][:k]
    
    # Calculate DCG
    dcg = np.sum([y_true[idx] / np.log2(i + 2) for i, idx in enumerate(pred_indices)])
    
    # Calculate ideal DCG
    ideal_indices = np.argsort(y_true)[::-1][:k]
    idcg = np.sum([y_true[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_indices)])
    
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(y_true, y_pred, k):
    """Modified Precision@k implementation."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Get top k indices
    indices = np.argsort(y_pred)[::-1][:k]
    
    # Get actual top k indices
    actual_top_k = np.argsort(y_true)[::-1][:k]
    
    # Calculate precision
    true_positives = len(set(indices) & set(actual_top_k))
    return true_positives / k

def recall_at_k(y_true, y_pred, k):
    """Modified Recall@k implementation."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Get top k indices
    indices = np.argsort(y_pred)[::-1][:k]
    
    # Get actual top k indices
    actual_top_k = np.argsort(y_true)[::-1][:k]
    
    # Calculate recall
    true_positives = len(set(indices) & set(actual_top_k))
    return true_positives / len(actual_top_k)

def f1_score_at_k(y_true, y_pred, k):
    """Modified F1@k implementation."""
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

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

