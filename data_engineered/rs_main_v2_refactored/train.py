from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import DataPreprocessor
from model import HybridRecommender, EnhancedListNetLoss
import numpy as np
import torch
from tqdm import tqdm, tqdm_notebook
import sys
import os

# Add these near the top of the file, after imports
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 64
NUM_EPOCHS = 20
EMBEDDING_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-4
L2_LAMBDA = 1e-4

# Add at the top with other constants
NUMERICAL_COLS = [
    "age",
    "duration",
    "acousticness",
    "key",
    "mode",
    "speechiness",
    "instrumentalness",
    "liveness",
    "tempo",
    "time_signature",
    "explicit",
    "music_age",
    "energy_loudness",
    "dance_valence"
]

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

def prepare_tensor_data(features, target, device):
    """Modified to handle preprocessed data with explicit type conversion"""
    # Ensure numerical features are float32
    numerical_features = features[NUMERICAL_COLS].astype(np.float32).values
    
    return {
        "user_id": torch.tensor(features["user_id_encoded"].values, dtype=torch.long).to(device),
        "gender_ids": torch.tensor(features["gender_encoded"].values, dtype=torch.long).to(device),
        "genre_features": torch.tensor(
            features[[col for col in features.columns if col.startswith("genre_tfidf_")]].values,
            dtype=torch.float32,
        ).to(device),
        "artist_features": torch.tensor(
            features[[col for col in features.columns if col.startswith("artist_tfidf_")]].values,
            dtype=torch.float32,
        ).to(device),
        "music_features": torch.tensor(
            features[[col for col in features.columns if col.startswith("music_tfidf_")]].values,
            dtype=torch.float32,
        ).to(device),
        "numerical_features": torch.tensor(
            numerical_features,
            dtype=torch.float32
        ).to(device),
        "release_years": torch.tensor(
            features["release_year_encoded"].values,
            dtype=torch.long
        ).to(device),
        "target": torch.tensor(target.values, dtype=torch.float32).unsqueeze(1).to(device),
    }

def create_dataloader(tensor_data, batch_size=64, shuffle=True):
    """Helper function to create a DataLoader from tensor data."""
    try:
        dataset = TensorDataset(
            tensor_data["user_id"],
            tensor_data["artist_features"],
            tensor_data["gender_ids"],
            tensor_data["music_features"],
            tensor_data["genre_features"],
            tensor_data["numerical_features"],
            tensor_data["release_years"],
            tensor_data["target"],
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        print("Tensor shapes:")
        for key, tensor in tensor_data.items():
            print(f"{key}: {tensor.shape}")
        raise

def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        # Add progress bar for validation
        val_iterator = tqdm(dataloader, desc="Validation", leave=False)
        for batch in val_iterator:
            (
                batch_user_id,
                batch_artist_features,
                batch_gender_ids,
                batch_music_features,
                batch_genre_features,
                batch_numerical_features,
                batch_release_years,
                batch_target,
            ) = batch

            predictions = model(
                batch_user_id,
                batch_artist_features,
                batch_gender_ids,
                batch_music_features,
                batch_genre_features,
                batch_numerical_features,
                batch_release_years,
            )
            batch_loss, _ = criterion(
                predictions,
                batch_target,
                batch_genre_features,
                batch_artist_features,
                batch_music_features,
            )
            val_loss += batch_loss.item()
            val_iterator.set_postfix({"batch_loss": f"{batch_loss.item():.4f}"})
    return val_loss / len(dataloader)

def train(model, dataloader, criterion, optimizer, device, check_gradients=False):
    model.train()
    total_loss = 0.0
    # Add progress bar for training batches
    train_iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, batch in train_iterator:
        (
            batch_user_id,
            batch_artist_features,
            batch_gender_ids,
            batch_music_features,
            batch_genre_features,
            batch_numerical_features,
            batch_release_years,
            batch_target,
        ) = batch

        optimizer.zero_grad()
        predictions = model(
            batch_user_id,
            batch_artist_features,
            batch_gender_ids,
            batch_music_features,
            batch_genre_features,
            batch_numerical_features,
            batch_release_years,
        )

        loss, loss_details = criterion(
            predictions,
            batch_target,
            batch_genre_features,
            batch_artist_features,
            batch_music_features,
        )

        loss.backward()
        
        # Add gradient checking for first batch
        if check_gradients and batch_idx == 0:
            print("\nGradient information:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(f"{name}: No gradient!")
                    else:
                        print(f"{name}: {param.grad.mean():.4f} (mean), {param.grad.std():.4f} (std)")

        optimizer.step()
        total_loss += loss.item()

        # Update progress bar instead of print
        train_iterator.set_postfix({
            "listnet_loss": f"{loss_details['listnet_loss']:.4f}",
            "ils_penalty": f"{loss_details['ils_penalty']:.4f}",
            "total_loss": f"{loss_details['total_loss']:.4f}"
        })

    return total_loss / len(dataloader)

def main():
    try:
        # Create models directory at the start
        os.makedirs("models", exist_ok=True)
        
        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize the preprocessor
        preprocessor = DataPreprocessor(
            test_size=0.2,
            random_state=42,
            max_artist_features=3000,
            max_genre_features=20,
            max_music_features=5000,
        )

        # Load and preprocess data
        filepath = "../data/engineered_data.csv"

        # In main function, add error handling for data loading
        try:
            data = preprocessor.load_data(filepath)
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            raise

        if data.empty:
            raise ValueError("Loaded data is empty")

        # Ensure 'plays' column exists
        if "plays" not in data.columns:
            raise ValueError("Target column 'plays' not found in input data")

        data_encoded = preprocessor.encode_features_train(data)

        # Add plays column to encoded data if not present
        if "plays" not in data_encoded.columns:
            data_encoded["plays"] = data["plays"]

        # Ensure 'user_id' column exists in the encoded data
        if "user_id" not in data_encoded.columns:
            data_encoded["user_id"] = data["user_id"]
        features = preprocessor.feature_engineering(data_encoded)

        # Verify plays column exists before splitting
        if "plays" not in features.columns:
            raise ValueError("Target column 'plays' lost during preprocessing")

        # Correctly define music_feature_cols before splitting
        music_feature_cols = [
            col for col in features.columns if col.startswith("music_tfidf_")
        ]

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

        # Convert features to tensors
        train_tensors = prepare_tensor_data(train_features, train_target, device)
        val_tensors = prepare_tensor_data(val_features, val_target, device)
        test_tensors = prepare_tensor_data(test_features, test_target, device)

        # Create dataloaders
        train_dataloader = create_dataloader(
            train_tensors, batch_size=TRAIN_BATCH_SIZE, shuffle=True
        )
        val_dataloader = create_dataloader(
            val_tensors, batch_size=VAL_BATCH_SIZE, shuffle=False
        )

        # Add this debug code before model initialization:
        print("Feature dimensions:")
        print(f"Gender classes: {len(preprocessor.gender_encoder.classes_)}")
        print(
            f"Music TF-IDF features: {len([col for col in train_features.columns if col.startswith('music_tfidf_')])}"
        )
        print(
            f"Genre TF-IDF features: {len([col for col in train_features.columns if col.startswith('genre_tfidf_')])}"
        )
        print(
            f"Artist TF-IDF features: {len([col for col in train_features.columns if col.startswith('artist_tfidf_')])}"
        )
        print(
            f"Release year classes: {len(preprocessor.release_year_encoder.categories_[0])}"
        )

        # Add this debug code in train.py before model initialization
        music_feature_cols = [
            col for col in train_features.columns if col.startswith("music_tfidf_")
        ]
        print(f"Actual music features count: {len(music_feature_cols)}")
        print(f"First few music feature names: {music_feature_cols[:5]}")

        # Before model initialization, add these variables
        num_genres = len(
            [col for col in train_features.columns if col.startswith("genre_tfidf_")]
        )
        num_artist_features = len(
            [col for col in train_features.columns if col.startswith("artist_tfidf_")]
        )

        # Update model initialization
        model = HybridRecommender(
            num_users=len(preprocessor.user_id_encoder.classes_),
            num_genders=len(preprocessor.gender_encoder.classes_),
            num_music_items=len(music_feature_cols),
            num_genres=num_genres,
            num_artist_features=num_artist_features,
            num_numerical_features=len(NUMERICAL_COLS),
            num_release_years=len(preprocessor.release_year_encoder.categories_[0]),
            embedding_dim=EMBEDDING_DIM,
        ).to(device)

        # After model initialization
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        # Add this after model initialization but before training loop
        print("\nInitial model weights:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data.mean():.4f} (mean), {param.data.std():.4f} (std)")

        # Define optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

        # Define loss function
        criterion = EnhancedListNetLoss(k=10)

        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            save_path="models/best_model.pth",
        )
        val_losses = []

        # Training loop with progress bar
        epoch_iterator = tqdm(range(NUM_EPOCHS), desc="Epochs", position=0)
        for epoch in epoch_iterator:
            epoch_loss = train(model, train_dataloader, criterion, optimizer, device, check_gradients=(epoch == 0))
            val_loss = validate_model(model, val_dataloader, criterion, device)
            
            # Update epoch progress bar
            epoch_iterator.set_postfix({
                "train_loss": f"{epoch_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

            # Early Stopping check
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("\nEarly stopping triggered")
                break

            scheduler.step(val_loss)

        # Load the best model before evaluation
        try:
            model.load_state_dict(torch.load("models/best_model.pth",weights_only=True))
            model = model.to(device)
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
            print("Using last model state instead")

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(
                test_tensors["user_id"],
                test_tensors["artist_features"],
                test_tensors["gender_ids"],
                test_tensors["music_features"],
                test_tensors["genre_features"],
                test_tensors["numerical_features"],
                test_tensors["release_years"],
            )

        # Convert tensors to numpy arrays
        predictions_np = predictions.cpu().numpy().reshape(-1)
        test_target_np = test_tensors["target"].cpu().numpy().reshape(-1)
        test_user_ids = test_features["user_id"].values

        # Evaluate the model per user
        evaluate_model_per_user(predictions_np, test_target_np, test_user_ids, k=10)

        # Save the trained model
        torch.save(model.state_dict(), "models/model.pth")
        print("Model saved to 'models/model.pth'")
        print("Training complete!")

        # Add these print statements at key points
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Training samples: {len(train_dataloader.dataset)}")
        print(f"Validation samples: {len(val_dataloader.dataset)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # After loading data, add debug information
        # print("\nData types after loading:")
        # print(data.dtypes)
        
        # After preprocessing
        # print("\nNumerical features dtypes:")
        # for col in NUMERICAL_COLS:
        #     print(f"{col}: {train_features[col].dtype}")
        
        # Before tensor conversion
        # print("\nShape of numerical features:", train_features[NUMERICAL_COLS].shape)
        print("Any null values:", train_features[NUMERICAL_COLS].isnull().any().any())

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save the model in its current state
        torch.save(model.state_dict(), "models/interrupted_model.pth")
        print("Model saved to 'models/interrupted_model.pth'")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

def evaluate_model_per_user(predictions, test_targets, user_ids, k=10):
    """Evaluate model performance per user using ranking metrics."""
    # Convert user_ids from list of numpy arrays to tuple of values for hashing
    try:
        # Convert each numpy array to a tuple for hashing
        user_ids_hashable = [tuple(uid) for uid in user_ids]

        # Get unique users
        unique_users = list(set(map(tuple, user_ids)))

        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Add progress bar for user evaluation
        user_iterator = tqdm(unique_users, desc="Evaluating users")
        for user in user_iterator:
            # Create mask for current user
            user_mask = [tuple(uid) == user for uid in user_ids_hashable]
            user_mask = np.array(user_mask)

            user_pred = predictions[user_mask]
            user_true = test_targets[user_mask]

            if len(user_pred) > 0:
                relevance_threshold = np.percentile(user_true, 80)
                user_true_binary = (user_true >= relevance_threshold).astype(int)

                ndcg_scores.append(ndcg_at_k(user_true, user_pred, k))
                precision_scores.append(precision_at_k(user_true_binary, user_pred, k))
                recall_scores.append(recall_at_k(user_true_binary, user_pred, k))
                f1_scores.append(f1_score_at_k(user_true_binary, user_pred, k))

                # Update progress bar with current scores
                user_iterator.set_postfix({
                    "NDCG": f"{ndcg_scores[-1]:.4f}",
                    "F1": f"{f1_scores[-1]:.4f}"
                })

        # Calculate and print averages
        if ndcg_scores:
            print(f"Average NDCG@{k}: {np.mean(ndcg_scores):.4f}")
            print(f"Average Precision@{k}: {np.mean(precision_scores):.4f}")
            print(f"Average Recall@{k}: {np.mean(recall_scores):.4f}")
            print(f"Average F1-score@{k}: {np.mean(f1_scores):.4f}")
        else:
            print("No valid predictions found for evaluation")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print(f"user_ids shape: {user_ids.shape}, dtype: {user_ids.dtype}")
        print(f"predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        print(f"test_targets shape: {test_targets.shape}, dtype: {test_targets.dtype}")
        raise

    return {
        "ndcg": np.mean(ndcg_scores) if ndcg_scores else 0,
        "precision": np.mean(precision_scores) if precision_scores else 0,
        "recall": np.mean(recall_scores) if recall_scores else 0,
        "f1": np.mean(f1_scores) if f1_scores else 0,
    }

if __name__ == "__main__":
    main()

