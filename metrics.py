
import torch

def calculate_ndcg(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10):
    # Ensure predictions and targets are sorted by relevance
    _, indices = torch.sort(predictions, descending=True)
    sorted_targets = targets[indices]
    
    # Calculate DCG
    dcg = 0.0
    for i in range(min(k, len(sorted_targets))):
        dcg += (2 ** sorted_targets[i] - 1) / torch.log2(torch.tensor(i + 2.0))
    
    # Calculate IDCG
    sorted_targets, _ = torch.sort(targets, descending=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_targets))):
        idcg += (2 ** sorted_targets[i] - 1) / torch.log2(torch.tensor(i + 2.0))
    
    # Handle edge case where IDCG is zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg