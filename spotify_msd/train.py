import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import DataPreprocessor
from model import HybridRecommender, EnhancedListNetLoss
import numpy as np
import torch
from tqdm import tqdm
import sys
import psutil
import gc
import math
from sklearn.utils import shuffle

# Constants
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
NUM_WORKERS = 4  # Adjust based on your CPU cores
PIN_MEMORY = True
PREFETCH_FACTOR = 2
SAMPLE_SIZE = 1_000_000  # Adjust this based on your available memory

NUMERICAL_COLS = [
    "age",
    "gender",
    "duration_ms",
    "loudness",
    "instrumentalness",
    "liveness",
    "tempo",
    "energy",
    "danceability",
    "valence",
    "year",
]

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path="models/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

class SpotifyDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return (
            self.features["user_id_encoded"].iloc[idx],
            self.features[[col for col in self.features.columns if col.startswith("artist_tfidf_")]].iloc[idx].values,
            self.features["gender_encoded"].iloc[idx],
            self.features[[col for col in self.features.columns if col.startswith("music_tfidf_")]].iloc[idx].values,
            self.features[[col for col in self.features.columns if col.startswith("genre_tfidf_")]].iloc[idx].values,
            self.features[NUMERICAL_COLS].iloc[idx].values,
            self.features["release_year_encoded"].iloc[idx],
            self.target.iloc[idx]
        )

def create_dataloader(features, target, batch_size=64, shuffle=True):
    dataset = SpotifyDataset(features, target)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )

def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
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

        train_iterator.set_postfix({
            "listnet_loss": f"{loss_details['listnet_loss']:.4f}",
            "ils_penalty": f"{loss_details['ils_penalty']:.4f}",
            "total_loss": f"{loss_details['total_loss']:.4f}"
        })

    return total_loss / len(dataloader)

def ndcg_at_k(y_true, y_pred, k):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    pred_indices = np.argsort(y_pred)[::-1][:k]
    dcg = np.sum([y_true[idx] / np.log2(i + 2) for i, idx in enumerate(pred_indices)])

    ideal_indices = np.argsort(y_true)[::-1][:k]
    idcg = np.sum([y_true[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_indices)])

    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(y_true, y_pred, k):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    indices = np.argsort(y_pred)[::-1][:k]
    actual_top_k = np.argsort(y_true)[::-1][:k]

    true_positives = len(set(indices) & set(actual_top_k))
    return true_positives / k

def recall_at_k(y_true, y_pred, k):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    indices = np.argsort(y_pred)[::-1][:k]
    actual_top_k = np.argsort(y_true)[::-1][:k]

    true_positives = len(set(indices) & set(actual_top_k))
    return true_positives / len(actual_top_k)

def f1_score_at_k(y_true, y_pred, k):
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model_per_user(predictions, test_targets, user_ids, k=10):
    try:
        user_ids_hashable = [tuple(uid) for uid in user_ids]
        unique_users = list(set(map(tuple, user_ids)))

        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        user_iterator = tqdm(unique_users, desc="Evaluating users")
        for user in user_iterator:
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

                user_iterator.set_postfix({
                    "NDCG": f"{ndcg_scores[-1]:.4f}",
                    "F1": f"{f1_scores[-1]:.4f}"
                })

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

def main():
    try:
        print("Initial memory usage:")
        print_memory_usage()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        preprocessor = DataPreprocessor(
            test_size=0.2,
            random_state=42,
            max_artist_features=3000,
            max_genre_features=20,
            max_music_features=5000,
        )

        filepath = "../data/msd_full_data_processed.csv"
        print("Loading and preprocessing data in chunks...")
        chunk_size = SAMPLE_SIZE // 4
        data_chunks = []
        
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            chunk_encoded = preprocessor.encode_features_train(chunk)
            chunk_features = preprocessor.feature_engineering(chunk_encoded)
            data_chunks.append(chunk_features)
            print(f"Processed chunk of size {len(chunk)}")
            clear_memory()
        
        features = pd.concat(data_chunks, ignore_index=True)
        if len(features) > SAMPLE_SIZE:
            features = features.sample(n=SAMPLE_SIZE, random_state=42)
        
        print(f"Final dataset size: {len(features)}")
        
        train_size = int(0.7 * len(features))
        val_size = int(0.15 * len(features))
        
        indices = shuffle(range(len(features)), random_state=42)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_features = features.iloc[train_indices]
        train_target = train_features.pop('plays')
        val_features = features.iloc[val_indices]
        val_target = val_features.pop('plays')
        test_features = features.iloc[test_indices]
        test_target = test_features.pop('plays')
        
        train_dataloader = create_dataloader(
            train_features, train_target,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True
        )
        val_dataloader = create_dataloader(
            val_features, val_target,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False
        )
        
        model = HybridRecommender(embedding_dim=EMBEDDING_DIM).to(device)
        criterion = EnhancedListNetLoss(l2_lambda=L2_LAMBDA)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)

        best_val_loss = float("inf")
        ACCUMULATION_STEPS = 4  # Adjust based on memory constraints

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

            # Training
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Forward pass
                predictions = model(*[b.to(device) for b in batch[:-1]])
                loss = criterion(predictions, batch[-1].to(device))
                
                # Scale loss by accumulation steps
                loss = loss / ACCUMULATION_STEPS
                loss.backward()
                
                # Update weights every ACCUMULATION_STEPS batches
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * ACCUMULATION_STEPS

            avg_train_loss = epoch_loss / len(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # Validation
            avg_val_loss = validate_model(model, val_dataloader, criterion, device)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")

            # Step the scheduler
            scheduler.step(avg_val_loss)

            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "models/best_model.pth")

        # Evaluation on test set
        print("\nEvaluating on test set...")
        model.load_state_dict(torch.load("models/best_model.pth"))
        model.eval()

        test_dataloader = create_dataloader(
            test_features, test_target,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False
        )

        all_predictions = []
        all_targets = []
        all_user_ids = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
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
                    batch_user_id.to(device),
                    batch_artist_features.to(device),
                    batch_gender_ids.to(device),
                    batch_music_features.to(device),
                    batch_genre_features.to(device),
                    batch_numerical_features.to(device),
                    batch_release_years.to(device),
                )

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_target.numpy().flatten())
                all_user_ids.extend(batch_user_id.numpy())

        # Convert lists to numpy arrays for evaluation
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_user_ids = np.array(all_user_ids)

        # Evaluate model performance per user
        evaluate_model_per_user(all_predictions, all_targets, all_user_ids, k=10)

    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()