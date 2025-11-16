# Efficient LLM Sharding on Heterogeneous Clusters

This project explores **capability-aware sharding** for large language models (LLMs) in environments where GPUs have **different compute speeds or memory capacities**.  
By profiling transformer layers and assigning heavier blocks to faster devices, the approach aims to **improve throughput** and **reduce idle time** in distributed training and inference.

---

## 🚀 Features
- Implements **three sharding modes**:
  1. **Uniform** – equal layer split across devices.  
  2. **Hetero** – static ratio based on relative device speed.  
  3. **Hetero Profiled** – uses measured per-layer cost for mapping.  
- Built on **PyTorch** and **Hugging Face Transformers (GPT-2 Small)**.  
- Simulates heterogeneity using GPU + CPU in Google Colab.  
- Generates JSON logs, CSV summaries, and throughput plots.  
- Ready to extend to **multi-GPU heterogeneous clusters**.

---

## 🧩 Repository Structure
efficient-llm-sharding/
│
├── src/ # Core Python modules
│ ├── sharded_module.py # GPT-2 wrapper with custom layer-to-device routing
│ ├── mapping.py # Uniform, Hetero, and Profiled layer mappings
│ └── train_eval.py # Benchmark loop and metric collection
│
├── scripts/
│ ├── run_benchmark.py # CLI runner for benchmarking modes
│ └── plot_results.py # Generates bar charts from JSON results
│
├── results/ # Output JSONs, CSVs, and figures
├── notebooks/
│ └── heterollm.ipynb # Main Colab notebook for running experiments
├── paper/
│ └── Efficient_Shards_LLMs.pdf
├── slides/
│ └── Efficient_Shards_Presentation.pptx
└── README.md


## ⚙️ Setup
Run in **Google Colab** or a local Python 3.10+ environment with PyTorch installed.

```bash
# Create environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio transformers datasets accelerate matplotlib rich tabulate fairscale

Usage
1️⃣ Run Benchmarks
python scripts/run_benchmark.py --mode both --samples 256 --seq-len 256
python scripts/run_benchmark.py --mode hetero_profiled --samples 256 --seq-len 256

2️⃣ Plot Results
python scripts/plot_results.py results --modes uniform hetero hetero_profiled --out results/tps_all_modes.png

3️⃣ Example Output
{
  "uniform": { "tokens_per_sec": 13494 },
  "hetero":  { "tokens_per_sec": 13695 },
  "hetero_profiled": { "tokens_per_sec": 12667 }
}


➡️ ~1.5 % throughput improvement for capability-aware sharding.

📊 Results Summary
Mode	Mean TPS	Δ vs Uniform
Uniform	13,494	—
Hetero	13,695	+1.49 %
Hetero Profiled	12,667	−6.13 %
🧩 Future Work

Dynamic layer rebalancing based on live throughput.

Multi-GPU tests on real heterogeneous clusters (e.g., A100 + T4).

Integrate memory usage and activation cost into cost model.
