import unittest
import torch
from src.centralkv.algorithm.stream_pagerank import StreamPageRank

class TestStreamPageRank(unittest.TestCase):
    def setUp(self):
        self.pagerank = StreamPageRank(lambda_val=0.5)
        
    def test_score_update(self):
        # 3 nodes: 0, 1, 2
        current_scores = torch.tensor([[1.0, 1.0, 1.0]])
        
        # New token (idx 2) attends to 0 and 1
        # Weights: 0.6 to node 0, 0.4 to node 1
        attn_weights = torch.tensor([[0.6, 0.4]])
        neighbor_indices = torch.tensor([[0, 1]])
        new_token_idx = 2
        
        updated = self.pagerank.update_scores(
            current_scores, attn_weights, new_token_idx, neighbor_indices
        )
        
        # Boost = 1 + 0.5 * score[2] = 1 + 0.5 * 1.0 = 1.5
        # Delta 0 = 0.6 * 1.5 = 0.9
        # Delta 1 = 0.4 * 1.5 = 0.6
        # New 0 = 1.0 + 0.9 = 1.9
        # New 1 = 1.0 + 0.6 = 1.6
        
        self.assertAlmostEqual(updated[0, 0].item(), 1.9)
        self.assertAlmostEqual(updated[0, 1].item(), 1.6)

if __name__ == '__main__':
    unittest.main()





