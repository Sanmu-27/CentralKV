import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_attention_heatmap(attention_matrix, tokens=None, title="Attention Heatmap", save_path="heatmap.png"):
    """
    Plots a heatmap of the attention matrix.
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.cpu().detach().numpy()
        
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, cmap="viridis", square=True)
    
    if tokens:
        # Only label some ticks if sequence is long
        step = max(1, len(tokens) // 50)
        plt.xticks(np.arange(0, len(tokens), step), tokens[::step], rotation=90, fontsize=8)
        plt.yticks(np.arange(0, len(tokens), step), tokens[::step], rotation=0, fontsize=8)
        
    plt.title(title)
    plt.xlabel("Key Token (Source)")
    plt.ylabel("Query Token (Target)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_eviction_dynamics(step_indices, cache_sizes, bridge_retention_rates, save_path="eviction_dynamics.png"):
    """
    Plots how cache size and bridge retention change over time.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cache Size', color=color)
    ax1.plot(step_indices, cache_sizes, color=color, linestyle='--', label='Cache Size')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Bridge Retention Rate (%)', color=color)  
    ax2.plot(step_indices, bridge_retention_rates, color=color, linewidth=2, label='Bridge Retention')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("CentralKV: Cache Dynamics vs. Structural Preservation")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()





