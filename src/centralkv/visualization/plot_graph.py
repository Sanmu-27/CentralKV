import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_graph(attention_matrix, tokens, threshold=0.1, save_path="attention_graph.png"):
    """
    Visualizes the sparse attention graph.
    
    Args:
        attention_matrix (torch.Tensor or np.ndarray): (SeqLen, SeqLen) attention weights
        tokens (List[str]): List of token strings
        threshold (float): Minimum attention weight to visualize an edge
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.cpu().numpy()
        
    G = nx.DiGraph()
    seq_len = len(tokens)
    
    # Add nodes
    for i in range(seq_len):
        G.add_node(i, label=tokens[i])
        
    # Add edges
    for i in range(seq_len): # Source (Query)
        for j in range(seq_len): # Target (Key)
            weight = attention_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
                
    # Plotting
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
    
    # Draw edges with varying thickness
    edges = G.edges(data=True)
    weights = [d['weight'] * 5 for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, arrowstyle='->', arrowsize=10)
    
    # Draw labels
    labels = {i: tokens[i] for i in range(seq_len)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("CentralKV Attention Graph Topology")
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {save_path}")

def plot_centrality_distribution(scores, save_path="centrality_dist.png"):
    """Plots the distribution of centrality scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Token Centrality Scores")
    plt.xlabel("Centrality Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()





