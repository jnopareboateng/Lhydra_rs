from dataclasses import dataclass
from typing import Dict, Tuple
from collections import defaultdict
from preprocessing import DataPreprocessor
from torch.utils.data import DataLoader

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import HybridRecommender, EnhancedListNetLoss
import torch
import numpy as np


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    batch_size: int = 256
    val_batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    l2_lambda: float = 1e-4


class MetricsTracker:
    """Handles metric computation and tracking"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, metric_dict: Dict[str, float]):
        for key, value in metric_dict.items():
            self.metrics[key].append(value)

    def get_average(self, metric_name: str) -> float:
        return np.mean(self.metrics[metric_name])

    def reset(self):
        self.metrics.clear()


class BatchProcessor:
    """Handles batch data processing"""

    def __init__(self, device: torch.device):
        self.device = device

    def process_batch(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        """Convert batch tuple to dictionary of tensors"""
        print(f"Batch length: {len(batch)}")
        print(f"Batch contents: {[type(b) for b in batch]}")

        # Safeguard against missing data
        if len(batch) < 8:
            raise ValueError(
                f"Expected 8 elements in batch, got {len(batch)}. Check DataPreprocessor output."
            )

        return {
            "user_id_hashed": batch[0].to(self.device),
            "artist_features": batch[1].to(self.device),
            "gender_ids": batch[2].to(self.device),
            "music_features": batch[3].to(self.device),
            "genre_features": batch[4].to(self.device),
            "numerical_features": batch[5].to(self.device),
            "release_years": batch[6].to(self.device),
            "target": batch[7].to(self.device),
        }


class ModelTrainer:
    """Handles model training and validation"""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.batch_processor = BatchProcessor(device)
        self.metrics_tracker = MetricsTracker()

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        self.metrics_tracker.reset()

        for batch in train_dataloader:
            loss = self._train_step(batch)
            self.metrics_tracker.update({"train_loss": loss})

        return self.metrics_tracker.get_average("train_loss")

    def _train_step(self, batch: Tuple) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        batch_data = self.batch_processor.process_batch(batch)

        # Remove target from the input data
        target = batch_data.pop("target")  # Extract and remove target

        # Forward pass with remaining features
        predictions = self.model(**batch_data)

        # Calculate loss using the target - unpack tuple
        total_loss, losses_dict = self.criterion(predictions, {"plays": target})

        # Backward pass on total_loss tensor
        total_loss.backward()
        self.optimizer.step()

        # Return the loss value
        return total_loss.item()

    def validate(self, val_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        self.metrics_tracker.reset()

        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = self.batch_processor.process_batch(batch)
                target = batch_data.pop("target")
                predictions = self.model(**batch_data)
                val_loss, losses_dict = self.criterion(predictions, {"plays": target})
                self.metrics_tracker.update({"val_loss": val_loss.item()})

        return {"val_loss": self.metrics_tracker.get_average("val_loss")}


class RankingMetrics:
    """Handles ranking metric computations"""

    @staticmethod
    def compute_metrics(
        predictions: np.ndarray, targets: np.ndarray, k: int
    ) -> Dict[str, float]:
        """Compute all ranking metrics at once"""
        relevance_threshold = np.percentile(targets, 80)
        binary_targets = (targets >= relevance_threshold).astype(int)

        return {
            "ndcg": RankingMetrics._ndcg_at_k(targets, predictions, k),
            "precision": RankingMetrics._precision_at_k(binary_targets, predictions, k),
            "recall": RankingMetrics._recall_at_k(binary_targets, predictions, k),
            "f1": RankingMetrics._f1_score_at_k(binary_targets, predictions, k),
        }

    @staticmethod
    def _ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        indices = np.argsort(y_pred)[::-1][:k]
        dcg = np.sum([y_true[idx] / np.log2(i + 2) for i, idx in enumerate(indices)])
        ideal_indices = np.argsort(y_true)[::-1][:k]
        idcg = np.sum(
            [y_true[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_indices)]
        )
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        return np.sum(y_true[np.argsort(y_pred)[::-1][:k]]) / k

    @staticmethod
    def _recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        return np.sum(y_true[np.argsort(y_pred)[::-1][:k]]) / np.sum(y_true)

    @staticmethod
    def _f1_score_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        precision = RankingMetrics._precision_at_k(y_true, y_pred, k)
        recall = RankingMetrics._recall_at_k(y_true, y_pred, k)
        return 2 * (precision * recall) / (precision + recall)

    # ... [Similar implementations for precision, recall, and f1]


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


def evaluate_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            batch_data = BatchProcessor(device).process_batch(batch)
            pred = model(**batch_data)["plays"]
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch_data["target"].cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    return RankingMetrics.compute_metrics(predictions, targets, k)


def main():
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        test_size=0.2,
        random_state=42,
        max_artist_features=3995,
        max_genre_features=21,
        max_music_features=5784,
    )

    # Load and preprocess data
    train_data, val_data, test_data = preprocessor.load_and_process_data()

    # Create dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_data, batch_size=config.val_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.val_batch_size)

    # Initialize model
    model = HybridRecommender(
        num_genders=2,
        num_music_items=preprocessor.max_music_features,
        num_genres=preprocessor.max_genre_features,
        num_artist_features=preprocessor.max_artist_features,
        num_numerical_features=5,  # Adjust based on your features
        num_release_years=100,  # Adjust based on your data
        user_id_hashed_dim=32,  # Adjust based on your hashing
    ).to(device)

    # Initialize optimizer and trainer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    trainer = ModelTrainer(
        model=model,
        criterion=EnhancedListNetLoss(k=10),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        ),
        config=config,
        device=device,
    )

    # Training loop
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    for epoch in range(config.num_epochs):
        train_loss = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.validate(val_dataloader)

        print(f"Epoch [{epoch+1}/{config.num_epochs}]:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")

        trainer.scheduler.step(val_metrics["val_loss"])
        early_stopping(val_metrics["val_loss"], trainer.model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Final evaluation
    test_metrics = evaluate_model(trainer.model, test_dataloader, device)
    print("\nTest Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
