import torch
from ..algorithm.stream_pagerank import StreamPageRank
from ..algorithm.sparse_graph import AttentionGraph

class EvictionPolicy:
    def update(self, scores, attention_logits, current_len):
        raise NotImplementedError

    def select_survivors(self, scores, budget, window_size, sys_window):
        raise NotImplementedError

class CentralKVPolicy(EvictionPolicy):
    def __init__(self, k_sparsity=8, lambda_val=0.5):
        self.pagerank = StreamPageRank(lambda_val=lambda_val)
        self.graph = AttentionGraph(max_size=100000, k_sparsity=k_sparsity)
        
    def update(self, scores, attention_logits, current_len):
        # 1. Construct Sparse Graph Edges (Top-K)
        # attention_logits: (Batch, Heads, 1, TotalSeqLen)
        weights, indices = self.graph.select_topk_edges(attention_logits, current_len)
        
        # 2. Update Stream-PageRank
        # We need to handle the dimension matching carefully here
        # scores: (Batch, Heads, TotalSeqLen)
        new_scores = self.pagerank.update_scores(
            current_scores=scores,
            attention_weights=weights,
            new_token_idx=current_len - 1,
            neighbor_indices=indices
        )
        return new_scores

    def select_survivors(self, scores, budget, window_size, sys_window):
        """
        Selects indices to keep:
        1. System Prompt (first W_sys tokens)
        2. Local Window (last W_local tokens)
        3. High Centrality Tokens (Top-K from the middle)
        """
        seq_len = scores.size(-1)
        
        # Indices
        all_indices = torch.arange(seq_len, device=scores.device)
        
        # Protected regions
        sys_indices = all_indices[:sys_window]
        local_indices = all_indices[-window_size:]
        
        # Middle region candidates
        middle_start = sys_window
        middle_end = seq_len - window_size
        
        if middle_end <= middle_start:
            # Overlap or empty middle, keep everything
            return all_indices.unsqueeze(0).unsqueeze(0).expand(scores.size(0), scores.size(1), -1)
            
        middle_scores = scores[..., middle_start:middle_end]
        
        # Calculate how many to keep in middle
        keep_middle_count = budget - sys_window - window_size
        if keep_middle_count <= 0:
             return torch.cat([sys_indices, local_indices]) # Fallback
             
        # Select top scores in middle
        _, top_middle_indices_rel = torch.topk(middle_scores, k=keep_middle_count, dim=-1)
        top_middle_indices = top_middle_indices_rel + middle_start
        
        # Merge and sort
        # Handle both 2D (batch, seq) and 3D (batch, heads, seq) cases
        if top_middle_indices.ndim == 2:
            # Shape: (batch, k)
            bsz, _ = top_middle_indices.shape
            sys_indices_expanded = sys_indices.view(1, -1).expand(bsz, -1)
            local_indices_expanded = local_indices.view(1, -1).expand(bsz, -1)
        elif top_middle_indices.ndim == 3:
            # Shape: (batch, heads, k)
            bsz, heads, _ = top_middle_indices.shape
            sys_indices_expanded = sys_indices.view(1, 1, -1).expand(bsz, heads, -1)
            local_indices_expanded = local_indices.view(1, 1, -1).expand(bsz, heads, -1)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {top_middle_indices.ndim}")
        
        combined = torch.cat([
            sys_indices_expanded, 
            top_middle_indices, 
            local_indices_expanded
        ], dim=-1)
        
        survivors, _ = torch.sort(combined, dim=-1)
        return survivors

class H2OPolicy(EvictionPolicy):
    """
    Baseline H2O: Accumulates attention scores (In-Degree).
    Equivalent to CentralKV with lambda=0 and dense edges.
    """
    def update(self, scores, attention_logits, current_len):
        # Simple accumulation of attention weights
        attn_weights = torch.softmax(attention_logits.squeeze(-2), dim=-1)
        return scores + attn_weights
        
    def select_survivors(self, scores, budget, window_size, sys_window):
        # Logic is identical to CentralKV, just the score source is different
        # (Accumulated Attention vs PageRank)
        return CentralKVPolicy().select_survivors(scores, budget, window_size, sys_window)



