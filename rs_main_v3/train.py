import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import DataPreprocessor
from model import HybridRecommender, ListNetLoss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path="models/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")  # Initialize with infinity
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:  # Changed condition
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
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
    filepath = "./data/cleaned_modv2.csv"
    data = preprocessor.load_data(filepath)

    # Ensure 'plays' column exists
    if "plays" not in data.columns:
        raise ValueError("Target column 'plays' not found in input data")

    data_encoded = preprocessor.encode_features(data)

    # Add plays column to encoded data if not present
    if "plays" not in data_encoded.columns:
        data_encoded["plays"] = data["plays"]

    features = preprocessor.feature_engineering(data_encoded)

    # Verify plays column exists before splitting
    if "plays" not in features.columns:
        raise ValueError("Target column 'plays' lost during preprocessing")

    # Correctly define music_feature_cols before splitting
    music_feature_cols = [col for col in features.columns if col.startswith('music_tfidf_')]

    (
        train_features,
        val_features,
        test_features,
        train_target,
        val_target,
        test_target,
    ) = preprocessor.split_data(features)
    # Save preprocessors
    preprocessor.save_preprocessors(directory="models/")

    # Add after preprocessing step
    train_user_id_hashed = torch.tensor(
        np.stack(train_features["user_id_hashed"].tolist()), dtype=torch.float
    ).to(device)

    train_gender_ids = torch.tensor(
        train_features["gender_encoded"].values, dtype=torch.long
    ).to(device)

    train_genre_features = torch.tensor(
        train_features[
            [col for col in train_features.columns if col.startswith("genre_tfidf_")]
        ].values,
        dtype=torch.float,
    ).to(device)

    train_artist_features = torch.tensor(
        train_features[
            [col for col in train_features.columns if col.startswith("artist_tfidf_")]
        ].values,
        dtype=torch.float,
    ).to(device)

    # Get correct music feature columns
    music_feature_cols = [col for col in train_features.columns if col.startswith('music_tfidf_')]

    # Use the corrected music_feature_cols for all datasets
    train_music_features = torch.tensor(
        train_features[music_feature_cols].values,
        dtype=torch.float
    ).to(device)

    val_music_features = torch.tensor(
        val_features[music_feature_cols].values,
        dtype=torch.float
    ).to(device)

    test_music_features = torch.tensor(
        test_features[music_feature_cols].values,
        dtype=torch.float
    ).to(device)

    train_numerical_features = torch.tensor(
        train_features[
            [
                "age",
                "duration",
                "acousticness",
                "danceability",
                "energy",
                "key",
                "loudness",
                "mode",
                "speechiness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "time_signature",
                "explicit",
            ]
        ].values,
        dtype=torch.float,
    ).to(device)
    train_release_years = torch.tensor(
        train_features["release_year_encoded"].values, dtype=torch.long
    ).to(device)
    train_target = (
        torch.tensor(train_target.values, dtype=torch.float).unsqueeze(1).to(device)
    )

    val_user_id_hashed = torch.tensor(
        np.stack(val_features["user_id_hashed"].tolist()), dtype=torch.float
    ).to(device)

    val_gender_ids = torch.tensor(
        val_features["gender_encoded"].values, dtype=torch.long
    ).to(device)

    val_genre_features = torch.tensor(
        val_features[
            [col for col in val_features.columns if col.startswith("genre_tfidf_")]
        ].values,
        dtype=torch.float,
    ).to(device)

    val_artist_features = torch.tensor(
        val_features[
            [col for col in val_features.columns if col.startswith("artist_tfidf_")]
        ].values,
        dtype=torch.float,
    ).to(device)

    val_numerical_features = torch.tensor(
        val_features[
            [
                "age",
                "duration",
                "acousticness",
                "danceability",
                "energy",
                "key",
                "loudness",
                "mode",
                "speechiness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "time_signature",
                "explicit",
            ]
        ].values,
        dtype=torch.float,
    ).to(device)
    val_release_years = torch.tensor(
        val_features["release_year_encoded"].values, dtype=torch.long
    ).to(device)
    val_target = (
        torch.tensor(val_target.values, dtype=torch.float).unsqueeze(1).to(device)
    )

    test_user_id_hashed = torch.tensor(
        np.stack(test_features["user_id_hashed"].tolist()), dtype=torch.float
    ).to(device)

    test_gender_ids = torch.tensor(
        test_features["gender_encoded"].values, dtype=torch.long
    ).to(device)

    test_genre_features = torch.tensor(
        test_features[  # Changed from train_features to test_features
            [col for col in test_features.columns if col.startswith("genre_tfidf_")]
        ].values,
        dtype=torch.float,
    ).to(device)

    test_artist_features = torch.tensor(
        test_features[
            [col for col in test_features.columns if col.startswith("artist_tfidf_")]
        ].values,
        dtype=torch.float,
    ).to(device)

    test_numerical_features = torch.tensor(
        test_features[
            [
                "age",
                "duration",
                "acousticness",
                "danceability",
                "energy",
                "key",
                "loudness",
                "mode",
                "speechiness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "time_signature",
                "explicit",
            ]
        ].values,
        dtype=torch.float,
    ).to(device)
    test_release_years = torch.tensor(
        test_features["release_year_encoded"].values, dtype=torch.long
    ).to(device)
    test_target = (
        torch.tensor(test_target.values, dtype=torch.float).unsqueeze(1).to(device)
    )

    # print(f"Max user_id: {train_user_ids.max().item()}")
    print(f"Max artist_id: {train_artist_features.max().item()}")
    # print(f"Max genre_id: {train_genre_ids.max().item()}")

    # Create validation tensors and DataLoader
    val_dataset = TensorDataset(
        val_user_id_hashed,
        val_artist_features,
        val_gender_ids.long(),  # Convert to long here
        val_music_features,
        val_genre_features,
        val_numerical_features,
        val_release_years.long(),  # Convert to long here
        val_target,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Add this debug code before model initialization:
    print("Feature dimensions:")
    print(f"Gender classes: {len(preprocessor.gender_encoder.classes_)}")
    print(f"Music TF-IDF features: {len([col for col in train_features.columns if col.startswith('music_tfidf_')])}")
    print(f"Genre TF-IDF features: {len([col for col in train_features.columns if col.startswith('genre_tfidf_')])}")
    print(f"Artist TF-IDF features: {len([col for col in train_features.columns if col.startswith('artist_tfidf_')])}")
    # print(f"Numerical features: 15")
    print(f"Release year classes: {len(preprocessor.release_year_encoder.classes_)}")

    # Correct the feature count retrieval
    num_music_items = len([col for col in train_features.columns if col.startswith("music_tfidf_")])
    num_genres = len([col for col in train_features.columns if col.startswith("genre_tfidf_")])
    num_artist_features = len([col for col in train_features.columns if col.startswith("artist_tfidf_")])

    print(f"Music TF-IDF features: {num_music_items}")
    print(f"Genre TF-IDF features: {num_genres}")
    print(f"Artist TF-IDF features: {num_artist_features}")

    # Add this debug code in train.py before model initialization
    music_feature_cols = [col for col in train_features.columns if col.startswith('music_tfidf_')]
    print(f"Actual music features count: {len(music_feature_cols)}")
    print(f"First few music feature names: {music_feature_cols[:5]}")

    # Update model initialization
    model = HybridRecommender(
        num_genders=len(preprocessor.gender_encoder.classes_),
        num_music_items=len(music_feature_cols),  # Use actual count
        num_genres=num_genres,
        num_artist_features=num_artist_features,
        num_numerical_features=15,
        num_release_years=len(preprocessor.release_year_encoder.classes_),
        embedding_dim=64,
    ).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # Define loss function
    criterion = ListNetLoss(k=10)

    # Create DataLoader for batch processing
    dataset = TensorDataset(
        train_user_id_hashed,
        train_artist_features,
        train_gender_ids.long(),  # Convert to long here
        train_music_features,
        train_genre_features,
        train_numerical_features,
        train_release_years.long(),  # Convert to long here
        train_target,
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    early_stopping = EarlyStopping(
        patience=5, min_delta=0.001, save_path="models/best_model.pth"
    )
    val_losses = []

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Training phase
        for batch in dataloader:  # Change enumerate to direct iteration
            (
                batch_user_id_hashed,
                batch_artist_features,
                batch_gender_ids,
                batch_music_features,
                batch_genre_features,
                batch_numerical_features,
                batch_release_years,
                batch_target,
            ) = batch  # Unpack correctly
            optimizer.zero_grad()

            predictions = model(
                batch_user_id_hashed,
                batch_artist_features,
                batch_gender_ids,
                batch_music_features,
                batch_genre_features,
                batch_numerical_features,
                batch_release_years,
            )

            loss = criterion(predictions, batch_target)

            # Add L2 regularization
            l2_reg = torch.tensor(0.0).to(device)
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
                (
                    batch_user_id_hashed,
                    batch_artist_features,
                    batch_gender_ids,
                    batch_music_features,
                    batch_genre_features,
                    batch_numerical_features,
                    batch_release_years,
                    batch_targets,
                ) = batch
                predictions = model(
                    batch_user_id_hashed,
                    batch_artist_features,
                    batch_gender_ids,
                    batch_music_features,
                    batch_genre_features,
                    batch_numerical_features,
                    batch_release_years,
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

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}"
        )
        scheduler.step(val_loss)

    # Load the best model before evaluation
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(
            test_user_id_hashed,
            test_artist_features,
            test_gender_ids,
            test_music_features,   # Correct: Now passing test_music_features
            test_genre_features,   # Correct: Now passing test_genre_features
            test_numerical_features,
            test_release_years,
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
