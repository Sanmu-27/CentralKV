import torch
from typing import Tuple

class AttentionGraph:
    """
    Manages the sparse graph structure constructed from attention weights.
    
    Nodes: Tokens in KV Cache
    Edges: Directed attention links (Top-K)
    """
    def __init__(self, max_size: int, k_sparsity: int = 8):
        self.max_size = max_size
        self.k = k_sparsity
        # Adjacency list could be stored, but for Stream-PageRank we mainly need
        # to process updates on the fly. We store statistics if needed.
        self.num_nodes = 0
        
    def select_topk_edges(self, attention_logits: torch.Tensor, past_key_values_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the top-k incoming edges for the current query token.
        
        Args:
            attention_logits: (Batch, Heads, SeqLen, TotalSeqLen) - Raw attention scores
            past_key_values_length: int - length of past context
            
        Returns:
            topk_weights: (Batch, Heads, K)
            topk_indices: (Batch, Heads, K)
        """
        # We focus on the last token's attention (the new query)
        # Shape: (Batch, Heads, 1, TotalSeqLen)
        last_token_attn = attention_logits[..., -1:, :] 
        last_token_attn = last_token_attn.squeeze(-2) # (Batch, Heads, TotalSeqLen)
        
        # We only care about attention to the *past* (valid cache), excluding current token self-loop if desired,
        # but usually self-loop is implicitly handled by softmax.
        
        # Apply Softmax to get probability weights
        attn_weights = torch.softmax(last_token_attn, dim=-1)
        
        # Top-K selection
        # We select top-k from the entire context available so far
        k = min(self.k, attn_weights.size(-1))
        topk_weights, topk_indices = torch.topk(attn_weights, k=k, dim=-1)
        
        return topk_weights, topk_indices

    def prune_graph(self, retained_indices: torch.Tensor):
        """
        Updates internal graph state when tokens are evicted from cache.
        In the simplified Stream-PageRank, we don't need to explicitly remove edges
        because the score is an accumulated scalar.
        """
        self.num_nodes = retained_indices.size(1)

