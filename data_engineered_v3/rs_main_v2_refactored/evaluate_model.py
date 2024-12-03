import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from typing import Dict, List, Tuple
import json
import os
from train_v2 import HybridMusicRecommender, MusicRecommenderDataset
from torch.utils.data import DataLoader
import logging
from sklearn.preprocessing import LabelEncoder

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
        
    def _initialize_model(self) -> HybridMusicRecommender:
        """Initialize and load the model from checkpoint."""
        model = HybridMusicRecommender(
            num_users=len(self.encoders['user_encoder'].classes_),
            num_music=len(self.encoders['music_encoder'].classes_),
            num_artists=len(self.encoders['artist_encoder'].classes_),
            num_genres=len(self.encoders['genre_encoder'].classes_),
            num_numerical=14,  # Number of numerical features
            embedding_dim=self.config['embedding_dim'],
            layers=self.config['hidden_layers'],
            dropout=self.config['dropout']
        )
        
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
                true_values.extend(batch['plays'].cpu().numpy())
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
                true_values.extend(batch['plays'].cpu().numpy())
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
    
    def plot_prediction_distribution(self, save_dir: str = 'metrics'):
        """Plot the distribution of predictions vs true values."""
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['plays'].cpu().numpy())
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
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prediction_distribution.png'))
        plt.close()
        
    def plot_error_distribution(self, save_dir: str = 'metrics'):
        """Plot the distribution of prediction errors."""
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['plays'].cpu().numpy())
                predictions.extend(pred.cpu().numpy())
        
        errors = np.array(predictions) - np.array(true_values)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()
    
    def evaluate_top_k_recommendations(self, k: int = 10) -> Dict[str, float]:
        """Evaluate top-K recommendation metrics."""
        true_values = []
        predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch)
                true_values.extend(batch['plays'].cpu().numpy())
                predictions.extend(pred.cpu().numpy())
        
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        
        # Calculate NDCG@K
        def calculate_ndcg(y_true, y_pred, k):
            # Get indices of top k predicted items
            top_k_indices = np.argsort(y_pred)[-k:][::-1]  # Sort descending
            
            # Get relevance scores (true values) for top k items
            rel = y_true[top_k_indices]
            
            # Calculate DCG
            dcg = rel[0] + np.sum(rel[1:] / np.log2(np.arange(2, len(rel) + 1)))
            
            # Calculate IDCG
            ideal_rel = np.sort(y_true)[::-1][:k]  # Sort descending
            idcg = ideal_rel[0] + np.sum(ideal_rel[1:] / np.log2(np.arange(2, len(ideal_rel) + 1)))
            
            # Calculate NDCG
            if idcg == 0:
                return 0.0
            return dcg / idcg
        
        # Calculate metrics for each user's recommendations
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        
        # Calculate metrics
        ndcg = calculate_ndcg(true_values, predictions, k)
        
        # Calculate precision and recall
        top_k_pred = np.argsort(predictions)[-k:]
        top_k_true = np.argsort(true_values)[-k:]
        
        precision = len(np.intersect1d(top_k_pred, top_k_true)) / k
        recall = len(np.intersect1d(top_k_pred, top_k_true)) / min(k, len(top_k_true))
        
        return {
            'ndcg@10': float(ndcg),
            'precision@10': float(precision),
            'recall@10': float(recall)
        }
    
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
                    true_values.extend(batch['plays'].cpu().numpy())
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

def main():
    # Load test data
    test_data = pd.read_csv('/home/josh/Lhydra_rs/data_engineered_v3/rs_main_v2_refactored/data/test_data.csv')
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='checkpoints/best_model.pth',
        test_data=test_data,
        batch_size=32
    )
    
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

if __name__ == "__main__":
    main()

