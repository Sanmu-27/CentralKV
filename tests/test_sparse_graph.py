import unittest
import torch
from src.centralkv.algorithm.sparse_graph import AttentionGraph

class TestSparseGraph(unittest.TestCase):
    def setUp(self):
        self.graph = AttentionGraph(max_size=100, k_sparsity=2)
        
    def test_topk_selection(self):
        # Batch=1, Heads=1, Seq=1, Total=4
        # Logits favour indices 0 and 2
        logits = torch.tensor([[[[10.0, 1.0, 8.0, 2.0]]]])
        
        weights, indices = self.graph.select_topk_edges(logits, past_key_values_length=3)
        
        # Should pick indices 0 and 2
        self.assertTrue(0 in indices[0, 0])
        self.assertTrue(2 in indices[0, 0])
        self.assertEqual(indices.shape[-1], 2)

if __name__ == '__main__':
    unittest.main()





