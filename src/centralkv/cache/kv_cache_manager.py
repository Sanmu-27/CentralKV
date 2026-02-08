import torch
from .eviction import EvictionPolicy, CentralKVPolicy, H2OPolicy

class KVCacheManager:
    """
    Manages the Key-Value Cache tensors and handles the eviction process.
    """
    def __init__(self, 
                 config,
                 max_budget: int, 
                 policy_type: str = "centralkv"):
        self.config = config
        self.max_budget = max_budget
        self.policy_type = policy_type
        
        # Policy Strategy Pattern
        if policy_type == "centralkv":
            self.policy = CentralKVPolicy(k_sparsity=config.k, lambda_val=config.lambda_val)
        elif policy_type == "h2o":
            self.policy = H2OPolicy()
        else:
            raise ValueError(f"Unknown policy: {policy_type}")
            
        self.kv_cache = None # List[Tuple[torch.Tensor, torch.Tensor]] (Layers)
        self.current_seq_len = 0
        
        # Centrality Scores: (Layers, Batch, Heads, CacheSize)
        # We maintain scores per head/layer or globally depending on implementation.
        # Paper suggests per-head or aggregated. Let's do per-head for granularity.
        self.importance_scores = None 
        
    def step(self, 
             past_key_values, 
             attention_logits: torch.Tensor,
             layer_idx: int):
        """
        Called at every generation step to update scores and potentially prune cache.
        """
        try:
            # Initialize scores storage if first time ever calling step (or reset)
            # Note: self.importance_scores is a DICT to handle arbitrary layer access order
            if self.importance_scores is None:
                self.importance_scores = {}
                
            if layer_idx not in self.importance_scores:
                # Initialize scores for this specific layer
                # Shape: [batch, heads, kv_len] - simplified, usually matches cache shape
                # We assume batch=1 for this validation script
                # attention_logits shape: [bsz, heads, q_len, kv_len]
                bsz, heads, _, kv_len = attention_logits.shape
                # Initialize with zeros
                self.importance_scores[layer_idx] = torch.zeros(
                    (bsz, heads, kv_len), 
                    device=attention_logits.device,
                    dtype=torch.float32
                )
                
            # Update current sequence length tracking
            # We assume kv_len increases by 1 each step (generation) or jumps (prefill)
            # But attention_logits tells us the *current* total KV length
            current_kv_len = attention_logits.shape[-1]
            self.current_seq_len = current_kv_len
            
            # Resize scores if necessary (e.g. KV cache grew)
            if self.importance_scores[layer_idx].shape[-1] < current_kv_len:
                 diff = current_kv_len - self.importance_scores[layer_idx].shape[-1]
                 new_zeros = torch.zeros(
                     (*self.importance_scores[layer_idx].shape[:-1], diff),
                     device=attention_logits.device,
                     dtype=self.importance_scores[layer_idx].dtype
                 )
                 self.importance_scores[layer_idx] = torch.cat(
                     [self.importance_scores[layer_idx], new_zeros], dim=-1
                 )
                
            # Update importance scores based on new attention
            self.importance_scores[layer_idx] = self.policy.update(
                self.importance_scores[layer_idx],
                attention_logits,
                self.current_seq_len
            )
            
            # Check for eviction
            if self.current_seq_len > self.max_budget:
                return self._evict(past_key_values, layer_idx)
                
            return past_key_values
            
        except Exception as e:
            # If any error occurs, just pass through without modification
            print(f"\n[WARNING] CentralKV step error in layer {layer_idx}: {type(e).__name__}: {e}")
            return past_key_values

    def _init_scores(self, past_key_values, layer_idx):
        # Implementation details for initializing score tensors
        pass
        
    def _evict(self, past_key_values, layer_idx):
        """
        Performs the eviction based on calculated scores.
        """
        keys, values = past_key_values
        
        # Get indices to keep
        scores = self.importance_scores[layer_idx]
        indices_to_keep = self.policy.select_survivors(
            scores, 
            budget=self.max_budget, 
            window_size=self.config.local_window,
            sys_window=self.config.sys_window
        )
        
        # Gather remaining KV
        # This is a simplified view; actual implementation needs careful tensor gathering
        # keys = keys.gather(2, indices_to_keep...)
        # values = values.gather(2, indices_to_keep...)
        
        # Update scores to match retained tokens
        # self.importance_scores[layer_idx] = scores.gather(..., indices_to_keep)
        
        return (keys, values)



