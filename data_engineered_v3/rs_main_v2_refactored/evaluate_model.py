import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from typing import Dict, List, Tuple
import json
import os
from train_model import HybridMusicRecommender, MusicRecommenderDataset
from torch.utils.data import DataLoader
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str, test_data: pd.DataFrame, batch_size: int = 32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.test_data = test_data
        self.batch_size = batch_size
        
        # Load model and config
        torch.serialization.add_safe_globals([LabelEncoder])
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint['config']
        self.encoders = self.checkpoint['encoders']
        
        # Initialize model
        self.model = self._initialize_model()
        self.test_loader = self._prepare_data()
        
        # Create metrics directory with absolute path
        self.metrics_dir = os.path.join(os.path.dirname(model_path), 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def _initialize_model(self, custom_config: Dict = None) -> HybridMusicRecommender:
        """Initialize and load the model from checkpoint."""
        # Use custom config if provided, otherwise use default
        config = custom_config if custom_config else self.config
        
        model = HybridMusicRecommender(
            num_users=len(self.encoders['user_encoder'].classes_),
            num_music=len(self.encoders['music_encoder'].classes_),
            num_artists=len(self.encoders['artist_encoder'].classes_),
            num_genres=len(self.encoders['genre_encoder'].classes_),
            num_numerical=12,
            embedding_dim=config['embedding_dim'],
            layers=config['hidden_layers'],
            dropout=config['dropout']
        )
        
        # Only load state dict if using default config
        if not custom_config:
            model.load_state_dict(self.checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _prepare_data(self) -> DataLoader:
        """Prepare test data loader using saved encoders."""
        # Create a custom dataset for test data with the saved encoders
        test_dataset = MusicRecommenderDataset(
            self.test_data, 
            mode='test',
            encoders=self.encoders
        )
        
        logger.info(f"Prepared test dataset with {len(self.test_data)} samples")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate various performance metrics."""
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['playcount'].cpu().numpy())
                predictions.extend(pred.cpu().numpy())
        
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        
        metrics = {
            'mse': float(mean_squared_error(true_values, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(true_values, predictions))),
            'mae': float(mean_absolute_error(true_values, predictions)),
            'r2': float(r2_score(true_values, predictions))
        }
        
        # Calculate prediction distribution statistics
        metrics.update({
            'pred_mean': float(np.mean(predictions)),
            'pred_std': float(np.std(predictions)),
            'true_mean': float(np.mean(true_values)),
            'true_std': float(np.std(true_values))
        })
        
        return metrics
    
    def analyze_prediction_bias(self) -> Dict[str, float]:
        """Analyze prediction bias across different value ranges."""
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['playcount'].cpu().numpy())
                predictions.extend(pred.cpu().numpy())
        
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        
        # Calculate bias for different value ranges
        percentiles = np.percentile(true_values, [25, 50, 75])
        ranges = [
            (float('-inf'), percentiles[0]),
            (percentiles[0], percentiles[1]),
            (percentiles[1], percentiles[2]),
            (percentiles[2], float('inf'))
        ]
        
        bias_analysis = {}
        for i, (low, high) in enumerate(ranges):
            mask = (true_values >= low) & (true_values < high)
            if np.any(mask):
                bias = np.mean(predictions[mask] - true_values[mask])
                bias_analysis[f'bias_range_{i+1}'] = float(bias)
        
        return bias_analysis
    
    def plot_prediction_distribution(self, save_dir: str = None):
        """Plot the distribution of predictions vs true values."""
        if save_dir is None:
            save_dir = self.metrics_dir
            
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['playcount'].cpu().numpy())
                predictions.extend(pred.cpu().numpy())
        
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([true_values.min(), true_values.max()], 
                [true_values.min(), true_values.max()], 
                'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Prediction vs True Values')
        
        try:
            # Save plot with absolute path
            plot_path = os.path.join(save_dir, 'prediction_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved prediction distribution plot to: {plot_path}")
        except Exception as e:
            logger.error(f"Error saving prediction distribution plot: {str(e)}")
            
    def plot_error_distribution(self, save_dir: str = None):
        """Plot the distribution of prediction errors."""
        if save_dir is None:
            save_dir = self.metrics_dir
            
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['playcount'].cpu().numpy())
                predictions.extend(pred.cpu().numpy())
        
        errors = np.array(predictions) - np.array(true_values)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        
        try:
            plot_path = os.path.join(save_dir, 'error_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved error distribution plot to: {plot_path}")
        except Exception as e:
            logger.error(f"Error saving error distribution plot: {str(e)}")
    
    def evaluate_top_k_recommendations(self, k: int = 10) -> Dict[str, float]:
        """Evaluate top-K recommendation metrics."""
        user_metrics = []
        
        # Group by user to evaluate per-user recommendations
        for user_id in self.test_data['user_id'].unique():
            user_mask = self.test_data['user_id'] == user_id
            user_data = self.test_data[user_mask]
            
            # Skip users with too few interactions
            if len(user_data) < k:
                continue
            
            user_dataset = MusicRecommenderDataset(
                user_data,
                mode='test',
                encoders=self.encoders
            )
            user_loader = DataLoader(user_dataset, batch_size=len(user_data), shuffle=False)
            
            with torch.no_grad():
                batch = next(iter(user_loader))
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = self.model(batch).cpu().numpy()
                true_values = batch['playcount'].cpu().numpy()
                
                # Normalize predictions and true values to [0, 1] range
                true_values = (true_values - true_values.min()) / (true_values.max() - true_values.min() + 1e-8)
                predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-8)
                
                # Calculate metrics for this user
                top_k_pred_idx = np.argsort(predictions)[-k:][::-1]
                top_k_true_idx = np.argsort(true_values)[-k:][::-1]
                
                # Calculate NDCG
                dcg = self._calculate_dcg(true_values, top_k_pred_idx, k)
                idcg = self._calculate_dcg(true_values, top_k_true_idx, k)
                
                # Handle edge case where idcg is 0
                ndcg = dcg / idcg if idcg > 0 else 0.0
                
                # Calculate precision and recall
                relevant_items = set(top_k_true_idx)
                recommended_items = set(top_k_pred_idx)
                
                precision = len(relevant_items & recommended_items) / k
                recall = len(relevant_items & recommended_items) / len(relevant_items)
                
                user_metrics.append({
                    'ndcg': ndcg,
                    'precision': precision,
                    'recall': recall
                })
        
        # Average metrics across users
        avg_metrics = {
            'ndcg@10': float(np.mean([m['ndcg'] for m in user_metrics])),
            'precision@10': float(np.mean([m['precision'] for m in user_metrics])),
            'recall@10': float(np.mean([m['recall'] for m in user_metrics]))
        }
        
        return avg_metrics

    def _calculate_dcg(self, true_values: np.ndarray, indices: np.ndarray, k: int) -> float:
        """Helper method to calculate DCG with numerical stability."""
        relevance = true_values[indices[:k]]
        # Cap the relevance values to prevent overflow
        max_relevance = 10  # Set a reasonable maximum value
        relevance = np.clip(relevance, 0, max_relevance)
        
        # Use log2(rank + 1) directly instead of creating array
        gains = (2 ** relevance - 1) / np.log2(np.arange(2, len(relevance) + 2))
        return float(np.sum(gains))
    
    def evaluate_cold_start(self, min_interactions: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on cold-start scenarios.
        
        Args:
            min_interactions: Minimum number of interactions to consider a user/item as non-cold
        
        Returns:
            Dictionary containing metrics for different cold-start scenarios
        """
        # Get all unique users and items
        all_users = self.test_data['user_id'].unique()
        all_items = self.test_data['music_id'].unique()
        
        # Count interactions per user and item
        user_counts = self.test_data['user_id'].value_counts()
        item_counts = self.test_data['music_id'].value_counts()
        
        # Identify cold users and items
        cold_users = set(user_counts[user_counts < min_interactions].index)
        cold_items = set(item_counts[item_counts < min_interactions].index)
        
        # Create masks for different scenarios
        cold_user_mask = self.test_data['user_id'].isin(cold_users)
        cold_item_mask = self.test_data['music_id'].isin(cold_items)
        cold_user_warm_item = cold_user_mask & ~cold_item_mask
        warm_user_cold_item = ~cold_user_mask & cold_item_mask
        cold_both = cold_user_mask & cold_item_mask
        warm_both = ~cold_user_mask & ~cold_item_mask
        
        scenarios = {
            'cold_user_warm_item': cold_user_warm_item,
            'warm_user_cold_item': warm_user_cold_item,
            'cold_both': cold_both,
            'warm_both': warm_both
        }
        
        results = {}
        for scenario_name, mask in scenarios.items():
            if not any(mask):
                logger.warning(f"No samples found for scenario: {scenario_name}")
                continue
                
            scenario_data = self.test_data[mask].copy()
            
            # Create a temporary dataset and dataloader for this scenario
            scenario_dataset = MusicRecommenderDataset(
                scenario_data,
                mode='test',
                encoders=self.encoders
            )
            
            scenario_loader = DataLoader(
                scenario_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Collect predictions and true values
            true_values = []
            predictions = []
            
            with torch.no_grad():
                for batch in scenario_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    pred = self.model(batch)
                    true_values.extend(batch['playcount'].cpu().numpy())
                    predictions.extend(pred.cpu().numpy())
            
            true_values = np.array(true_values)
            predictions = np.array(predictions)
            
            # Calculate metrics
            metrics = {
                'count': len(true_values),
                'mse': float(mean_squared_error(true_values, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(true_values, predictions))),
                'mae': float(mean_absolute_error(true_values, predictions)),
                'r2': float(r2_score(true_values, predictions)),
                'pred_mean': float(np.mean(predictions)),
                'pred_std': float(np.std(predictions)),
                'true_mean': float(np.mean(true_values)),
                'true_std': float(np.std(true_values))
            }
            
            results[scenario_name] = metrics
            
            # Log results for this scenario
            logger.info(f"\n{scenario_name} Metrics (n={metrics['count']}):")
            for metric, value in metrics.items():
                if metric != 'count':
                    logger.info(f"{metric}: {value:.4f}")
        
        return results
    
    def save_evaluation_results(self, save_dir: str = 'metrics'):
        """Run all evaluations and save results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate all metrics
        results = {
            'basic_metrics': self.calculate_metrics(),
            'bias_analysis': self.analyze_prediction_bias(),
            'top_k_metrics': self.evaluate_top_k_recommendations(),
            'cold_start_metrics': self.evaluate_cold_start(min_interactions=5)
        }
        
        # Save results to JSON
        results_file = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Evaluation completed. Results saved to: {save_dir}")
        
        return results

    def tune_hyperparameters(self, param_grid: Dict[str, List], val_data: pd.DataFrame) -> Dict:
        """
        Tune hyperparameters using validation set.
        
        Args:
            param_grid: Dictionary of parameters to try
            val_data: Validation data
            
        Returns:
            Best parameters found
        """
        best_score = float('inf')
        best_params = None
        
        # Create validation dataset
        val_dataset = MusicRecommenderDataset(val_data, mode='test', encoders=self.encoders)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Try all parameter combinations
        for params in ParameterGrid(param_grid):
            # Create a new config with updated parameters
            current_config = self.config.copy()
            current_config.update(params)
            
            # Initialize model with current parameters
            self.model = self._initialize_model(custom_config=current_config)
            
            # Evaluate on validation set
            metrics = self.calculate_metrics()
            score = metrics['rmse']  # Use RMSE as scoring metric
            
            if score < best_score:
                best_score = score
                best_params = params
                logger.info(f"New best parameters found: {params} (RMSE: {score:.4f})")
        
        return best_params

def main():
    # Load test data and check for data compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_path = os.path.join(ROOT_DIR, 'data', 'test_data.csv')
    model_path = os.path.join(ROOT_DIR, 'data_engineered_v3', 'rs_main_v2_refactored', 'checkpoints', 'best_model.pth')
    
    test_data = pd.read_csv(test_path)
    logger.info(f"Loaded test data with {len(test_data)} samples")
    
    # Split test data into validation and test
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            test_data=test_data,
            batch_size=32
        )
        
        # Tune hyperparameters
        param_grid = {
            'embedding_dim': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3],
            'hidden_layers': [[128, 64], [256, 128, 64], [512, 256, 128]]
        }
        
        best_params = evaluator.tune_hyperparameters(param_grid, val_data)
        logger.info(f"Best parameters: {best_params}")
        
        # Run evaluation
        results = evaluator.save_evaluation_results()
        
        # Print summary
        logger.info("\nEvaluation Summary:")
        logger.info("Basic Metrics:")
        for metric, value in results['basic_metrics'].items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nTop-K Metrics:")
        for metric, value in results['top_k_metrics'].items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nBias Analysis:")
        for range_name, bias in results['bias_analysis'].items():
            logger.info(f"{range_name}: {bias:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()

