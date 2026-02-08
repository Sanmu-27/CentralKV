import torch
import torch.nn.functional as F

class StreamPageRank:
    """
    Implements the Stream-PageRank algorithm for incremental centrality updates.
    
    Equation: S(j)^(t) = S(j)^(t-1) + A_{tj} * (1 + lambda * S(t)^{(t-1)})
    """
    def __init__(self, lambda_val: float = 0.5, damping: float = 0.85):
        self.lambda_val = lambda_val
        self.damping = damping
        # Scores are stored in the CacheManager, this class handles the update logic
        
    def update_scores(self, 
                      current_scores: torch.Tensor, 
                      attention_weights: torch.Tensor, 
                      new_token_idx: int, 
                      neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        Updates the centrality scores based on the new attention vector.
        
        Args:
            current_scores: (Batch, CacheSize) - Current centrality scores
            attention_weights: (Batch, NumNeighbors) - Attention weights from new token to neighbors
            new_token_idx: int - Index of the new token in the cache (usually last)
            neighbor_indices: (Batch, NumNeighbors) - Indices of the attended tokens
            
        Returns:
            Updated scores tensor
        """
        batch_size = attention_weights.shape[0]
        # attention_weights shape can be (Batch, Heads, NumNeighbors) or (Batch, NumNeighbors)
        # We need to handle both cases or ensure we are calling it correctly.
        
        # If we have heads dimension, we flatten or treat as batch*heads
        if len(attention_weights.shape) == 3:
             bsz, heads, num_neighbors = attention_weights.shape
             # Treat batch*heads as the batch dimension for logic
             attention_weights = attention_weights.reshape(bsz * heads, num_neighbors)
             neighbor_indices = neighbor_indices.reshape(bsz * heads, num_neighbors)
             current_scores = current_scores.reshape(bsz * heads, -1)
             
             batch_size = bsz * heads
        
        num_neighbors = attention_weights.shape[-1]
        
        # 1. Get score of the current new token (query)
        # In a strict online setting, the new token S(t) starts at 0 or a base value.
        # We can approximate S(t) by its self-attention or inherit from average.
        # Here we assume S(t) is initialized to a base value (e.g. 1.0) before calling update.
        
        # Gather S(t) for each batch
        # shape: (Batch, 1)
        query_scores = current_scores.gather(1, torch.tensor([[new_token_idx]] * batch_size, device=current_scores.device))
        
        # 2. Compute the boost factor: (1 + lambda * S(t))
        boost = 1.0 + self.lambda_val * query_scores
        
        # 3. Calculate score delta for neighbors: A_{tj} * boost
        score_deltas = attention_weights * boost
        
        # 4. Scatter add the deltas to the current scores
        # We use scatter_add_ to update in-place or return new tensor
        updated_scores = current_scores.clone()
        updated_scores.scatter_add_(1, neighbor_indices, score_deltas)
        
        # Reshape back if we flattened
        if len(updated_scores.shape) == 2 and batch_size != updated_scores.shape[0]:
             # This happens if we manipulated batch_size but return logic assumes original shape?
             # Actually, if we flattened inputs, we return flattened output, and caller reshapes.
             pass
             
        return updated_scores

    def initialize_score(self, scores: torch.Tensor, idx: int, init_value: float = 1.0):
        """Initializes the score for a newly added token."""
        scores[:, idx] = init_value



