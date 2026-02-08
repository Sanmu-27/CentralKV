# CentralKV: Compressing Long-Context LLMs via Attention-Graph Centrality

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

> ⚠️ **Note:** This repository is a **preview release** submitted for peer review, focused on **functional verification**. The full codebase, including extensive baselines, optimized kernels, and scripts to reproduce the **full paper results**, will be released upon acceptance.

**CentralKV** is a training-free framework designed to address the memory bottlenecks of Long-Context Large Language Models (LLMs). By modeling the self-attention mechanism as a dynamic sparse graph, CentralKV identifies and retains "Structural Bridges"—tokens with high graph centrality that act as critical semantic connectors—ensuring coherent reasoning even under high compression rates.

---

## 🌟 Key Features

- **Graph-Theoretic Compression**  
  Moves beyond first-order statistics (e.g., accumulated attention in H2O) to capture higher-order topological importance via PageRank centrality.

- **Zero-Cost Construction**  
  Efficiently constructs the attention graph by reusing existing attention weights, avoiding the $O(N \cdot d)$ overhead of similarity-based methods like GraphKV.

- **Stream-PageRank Algorithm**  
  Implements an incremental, amortized $O(1)$ centrality update mechanism, making it suitable for real-time streaming inference.

- **Seamless Integration**  
  Plug-and-play compatibility with popular RoPE-based models (Llama-2, Llama-3, TinyLlama) via standard HuggingFace Transformers integration.

---

## 📂 Project Structure

The repository is organized to facilitate reproducibility and extension:

```text
.
├── benchmarks/         # Evaluation scripts (LongBench, Needle-in-a-Haystack)
├── configs/            # Configuration files for models and experiments
├── scripts/            # Shell scripts for batch experiment reproduction
├── src/
│   └── centralkv/
│       ├── algorithm/      # Core logic: Stream-PageRank & Sparse Graph
│       ├── analysis/       # Tools for profiling & graph topology export
│       ├── cache/          # KV Cache eviction policies (CentralKV, H2O)
│       ├── models/         # Model patches (Llama/TinyLlama support)
│       └── visualization/  # Analysis tools (Heatmaps, Graph Viz)
└── tests/              # Unit tests for algorithm verification
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch >= 2.0
- Transformers >= 4.36

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Usage Example

CentralKV can be easily integrated into existing pipelines using our `enable_centralkv_for_llama` patch:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.centralkv.models import enable_centralkv_for_llama

# 1. Load Model
model_id = "HuggingFaceTB/SmolLM-135M" # Or meta-llama/Llama-2-7b-chat-hf
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Enable Compression
# budget_ratio=0.2 retains 20% of the KV cache
enable_centralkv_for_llama(model, budget_ratio=0.2, sparsity_k=8)

# 3. Generate
inputs = tokenizer("Your long context here...", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
```

---

## 📊 Analysis & Visualization

We provide a comprehensive suite of tools to analyze the attention dynamics and verify the "Bridge Token" hypothesis.

### Attention Graph Visualization
Visualize the sparse attention topology to identify structural hubs:

```python
from src.centralkv.visualization import visualize_attention_graph

# Pass your attention matrix and tokens
visualize_attention_graph(matrix, tokens, save_path="graph_viz.png")
```

### Profiling
Measure latency and memory overhead overhead:

```python
from src.centralkv.analysis import CentralKVProfiler

profiler = CentralKVProfiler()
with profiler.timer("inference_step"):
    # ... generation loop ...
profiler.summarize()
```

### Graph Export
Export internal states to `.gexf` for analysis in tools like Gephi:

```python
from src.centralkv.analysis import GraphExporter

exporter = GraphExporter()
exporter.export_to_gexf(graph_data, step=100)
```

---

## 🧪 Experiments

### Needle In A Haystack (NIAH)
Verify long-context retrieval capabilities (Figure 3 in the paper).

**Note:** For fast verification during peer review, we default to using `SmolLM-135M` (a lightweight Llama architecture).

```bash
# Downloads SmolLM-135M to ./local_models and runs a grid search (Length 1k-32k)
python benchmarks/needle_in_haystack/run_niah.py
```

To run with the full `Llama-2-7b-chat` (requires HuggingFace authentication):

```bash
python benchmarks/needle_in_haystack/run_niah.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --cache_dir ./models
```

### LongBench Evaluation
Validate the compression logic on standard benchmarks.

> **Note:** The provided script runs a **mini-batch verification** to demonstrate functional correctness. Reproducing the full paper results (Table 2) requires the full Llama-2-7B/Llama-3-8B models and the complete LongBench dataset (approx. 200GB+).

```bash
bash scripts/run_longbench.sh --model llama3-8b --method centralkv
```

---

## ⚖️ License

This project is released under the MIT License.
