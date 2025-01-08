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
        # Get dimensions from encoder
        try:
            dims = self.encoders.get_dims()
            logger.info(f"Model dimensions: {dims}")
            
            model = HybridMusicRecommender(
                music_dims=dims['music_dims'],
                artist_dims=dims['artist_dims'],
                genre_dims=dims['genre_dims'],  # Must match TF-IDF size
                num_numerical=dims['num_numerical'],
                embedding_dim=self.config.get('embedding_dim', 64),
                layers=self.config.get('hidden_layers', [256, 128, 64]),
                dropout=self.config.get('dropout', 0.2)
            )
            
            # Load state dict with strict=False to allow partial loading
            if not custom_config:
                try:
                    model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
                    logger.info("Model state loaded successfully")
                except Exception as e:
                    logger.warning(f"Error loading model state: {str(e)}")
                    logger.warning("Initializing model with fresh weights")
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except KeyError as e:
            logger.error(f"Missing key in encoders: {str(e)}")
            raise
    
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
        """Improved top-K evaluation with proper NDCG calculation."""
        metrics = {'ndcg@10': 0.0, 'precision@10': 0.0, 'recall@10': 0.0}
        total_users = 0
        
        # Group by age and gender as proxy for user groups
        for (age, gender), group_data in self.test_data.groupby(['age', 'gender']):
            if len(group_data) < k:
                continue
                
            user_dataset = MusicRecommenderDataset(
                group_data,
                mode='test',
                encoders=self.encoders
            )
            user_loader = DataLoader(user_dataset, batch_size=len(group_data), shuffle=False)
            
            with torch.no_grad():
                batch = next(iter(user_loader))
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = self.model(batch).cpu().numpy()
                true_values = batch['playcount'].cpu().numpy()
                
                # Ensure predictions and true_values are 1-dimensional
                predictions = predictions.flatten()
                true_values = true_values.flatten()
                
                # Calculate NDCG
                top_k_pred_idx = np.argsort(predictions)[-k:][::-1]
                top_k_true_idx = np.argsort(true_values)[-k:][::-1]
                
                # Use sklearn's ndcg_score for more reliable calculation
                ndcg = ndcg_score(
                    true_values.reshape(1, -1),
                    predictions.reshape(1, -1),
                    k=k
                )
                
                # Calculate precision and recall
                relevant_items = set(top_k_true_idx)
                recommended_items = set(top_k_pred_idx)
                
                n_relevant_and_recommended = len(relevant_items & recommended_items)
                precision = n_relevant_and_recommended / k
                recall = n_relevant_and_recommended / len(relevant_items)
                
                metrics['ndcg@10'] += ndcg
                metrics['precision@10'] += precision
                metrics['recall@10'] += recall
                total_users += 1
        
        # Average metrics
        if total_users > 0:
            metrics = {k: v / total_users for k, v in metrics.items()}
        
        return metrics

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
        """Update to handle multidimensional predictions."""
        # Get unique items and their interaction counts
        item_counts = self.test_data['music'].value_counts()
        cold_items = set(item_counts[item_counts < min_interactions].index)
        
        # Create masks for different scenarios
        cold_item_mask = self.test_data['music'].isin(cold_items)
        warm_item_mask = ~cold_item_mask
        
        scenarios = {
            'cold_items': cold_item_mask,
            'warm_items': warm_item_mask
        }
        
        results = {}
        for scenario_name, mask in scenarios.items():
            if not any(mask):
                logger.warning(f"No samples found for scenario: {scenario_name}")
                continue
                
            scenario_data = self.test_data[mask].copy()
            
            # Create dataset and loader for this scenario
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
                    # Ensure predictions are 1-dimensional
                    if len(pred.shape) > 1:
                        pred = pred.squeeze()
                    # Convert to numpy array and flatten
                    pred_np = pred.cpu().numpy()
                    if isinstance(pred_np, np.ndarray):
                        predictions.extend(pred_np.flatten())
                    else:
                        predictions.append(float(pred_np))
                    true_values.extend(batch['playcount'].cpu().numpy().flatten())
            
            # Convert lists to numpy arrays
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
    # Update paths to match actual directory structure
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_path = os.path.join(ROOT_DIR, 'data','processed_data', 'test_data.csv')  # This will point to /home/martinson/Lhydra_rs/data/test_data.csv
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best_model.pth')
    
    # Validate paths exist
    if not os.path.exists(test_path):
        logger.error(f"Test data not found at: {test_path}")
        raise FileNotFoundError(f"Test data file not found at {test_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'latest_checkpoint.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint found in {os.path.dirname(model_path)}")

    # Print resolved paths
    logger.info(f"Using test data from: {test_path}")
    logger.info(f"Using model from: {model_path}")

    # Load test data and check for data compatibility
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

import unittest

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        # Mock model and data
        self.model = HybridMusicRecommender(...).to('cpu')
        self.test_data = pd.DataFrame({
            # ...populate with test data...
        })
        self.evaluator = ModelEvaluator('path/to/mock_model.pth', self.test_data, batch_size=2)

    def test_initialize_model(self):
        self.assertIsNotNone(self.evaluator.model)

    def test_calculate_metrics(self):
        metrics = self.evaluator.calculate_metrics()
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

    def test_analyze_prediction_bias(self):
        bias = self.evaluator.analyze_prediction_bias()
        self.assertIsInstance(bias, dict)

    def test_evaluate_cold_start(self):
        results = self.evaluator.evaluate_cold_start()
        self.assertIn('cold_items', results)
        self.assertIn('warm_items', results)

    def test_tune_hyperparameters(self):
        param_grid = {'embedding_dim': [32, 64], 'dropout': [0.1, 0.2]}
        val_data = pd.DataFrame({
            # ...populate with validation data...
        })
        best_params = self.evaluator.tune_hyperparameters(param_grid, val_data)
        self.assertIn('embedding_dim', best_params)
        self.assertIn('dropout', best_params)

