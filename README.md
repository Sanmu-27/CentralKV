# CentralKV: Compressing Long-Context LLMs via Attention-Graph Centrality

Accepted to **EMNLP 2026 Main Conference**.

Authors: Yusen Wu, Huang Yinjun, Jia Yee Tan, Hao Li, Rongfeng Guo, Liang Fan, Jiachen Luo, Guangyuan Dong, and Sike Xiang (corresponding author).

CentralKV is a training-free method for reducing the memory cost of long-context large language models. It models the causal attention pattern as a dynamic sparse graph and preserves structurally important bridge tokens during KV-cache compression. The implementation in this repository contains the algorithm, model integration, evaluation scripts, and unit tests used for the paper.

Project page: https://sanmu-27.github.io/CentralKV-/  
Paper: https://sanmu-27.github.io/CentralKV-/static/pdfs/CentralKV.pdf

---

## Method

- **Attention graph construction**  
  Reuses attention weights to construct a sparse causal graph and estimate higher-order token importance.

- **Stream-PageRank updates**  
  Maintains centrality scores incrementally during decoding without a separate training stage.

- **Cache preservation**  
  Protects the system prompt and recent local window, then evicts low-centrality tokens under a prescribed budget.

- **Model integration**  
  Provides a HuggingFace Transformers patch for supported RoPE-based causal language models.

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

## Getting started

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

## Analysis and visualization

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

## Experiments

### Needle In A Haystack (NIAH)
Verify long-context retrieval capabilities.

For a lightweight local check, the default configuration uses `SmolLM-135M`.

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

The provided script runs a compact verification. Reproducing the complete reported results requires the corresponding Llama models and the full LongBench dataset.

```bash
bash scripts/run_longbench.sh --model llama3-8b --method centralkv
```

---

## ⚖️ License

This project is released under the MIT License.

## Citation

```bibtex
@inproceedings{wu2026centralkv,
  title     = {CentralKV: Compressing Long-Context LLMs via Attention-Graph Centrality},
  author    = {Wu, Yusen and Yinjun, Huang and Tan, Jia Yee and Li, Hao and Guo, Rongfeng and Fan, Liang and Luo, Jiachen and Dong, Guangyuan and Xiang, Sike},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  url       = {https://sanmu-27.github.io/CentralKV-/}
}
```
