# SovereignLoRA Lab

**Local Edge LLM Fine-Tuning, Direct Preference Optimization (DPO), and Reinforcement Learning (GRPO) Engine.**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Python: >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/)
[![Hardware: Consumer Edge](https://img.shields.io/badge/hardware-8GB%20VRAM%20%2F%20Apple%20Silicon-brightgreen.svg)]()
[![Author: Ethan-da-Tech-Wizard](https://img.shields.io/badge/author-Ethan--da--Tech--Wizard-orange.svg)](https://github.com/Ethan-da-Tech-Wizard)

SovereignLoRA Lab is an open-source, production-ready framework for training, aligning, and deploying state-of-the-art language models on consumer hardware (such as 8GB VRAM NVIDIA RTX 3070 and Apple Silicon M-series).

```
                      ┌──────────────────────────────────────────────┐
                      │             SovereignLoRA Lab                │
                      └──────────────────────┬───────────────────────┘
                                             │
      ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
      ▼                      ▼                              ▼                      ▼
┌────────────┐        ┌──────────────┐              ┌──────────────┐        ┌──────────────┐
│  Dataset   │   ──>  │ SFT Trainer  │     ──>      │ DPO / GRPO   │   ──>  │ Multi-LoRA   │
│ Validation │        │ (Chunked CE) │              │ Verification │        │ SVD Fusion   │
└────────────┘        └──────────────┘              └──────────────┘        └──────┬───────┘
                                                                                   │
                                                                                   ▼
                                                                            ┌──────────────┐
                                                                            │ GGUF Quant   │
                                                                            │ Q4_K_M / Q8  │
                                                                            └──────────────┘
```

---

## Educational Foundations & Core Concepts (CIS 156 / Advanced Systems)

SovereignLoRA Lab is structured as both a high-assurance production framework and an educational laboratory for students, researchers, and systems engineers. Here is how each underlying system works:

### 1. Low-Rank Adaptation (LoRA)
* **The Problem**: Full fine-tuning of a 7-billion parameter language model requires updating and storing $7 \times 10^9$ 16-bit parameters ($\approx 14\text{ GB}$ of weights) plus optimizer states ($28\text{ GB}$ for AdamW) and activation memory, requiring data center GPUs with $80\text{ GB}$ VRAM.
* **The Mathematics**: LoRA freezes the original pre-trained weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and decomposes the update matrix $\Delta W$ into two low-rank matrices:
  $$\Delta W = \frac{\alpha}{r} (B \times A)$$
  where:
  - $A \in \mathbb{R}^{r \times d_{\text{in}}}$ projects the input down to low-rank dimension $r$ (initialized from Gaussian noise $\mathcal{N}(0, \sigma^2)$).
  - $B \in \mathbb{R}^{d_{\text{out}} \times r}$ projects the low-rank representation back up to output dimension (initialized to $0$ so training starts with $\Delta W = 0$).
  - $r \ll \min(d_{\text{in}}, d_{\text{out}})$ (e.g. $r = 16$), reducing trainable parameters by over $99\%$.
  - $\frac{\alpha}{r}$ is a constant scaling hyperparameter that stabilizes gradient updates when changing rank $r$.

### 2. Truncated SVD Multi-Adapter Rank Compression ($3r \to r$)
* **The Problem**: When combining multiple specialist adapters (e.g. Tool Use, Code Repair, and Reasoning), naive linear task arithmetic expands the effective rank: $\operatorname{rank}(\Delta W_1 + \Delta W_2 + \Delta W_3) \le 3r$. Loading multiple adapters simultaneously exhausts consumer GPU memory.
* **The Solution**: We compute the exact combined delta matrix and apply Truncated Singular Value Decomposition (SVD):
  $$\Delta W_{\text{merged}} = U \Sigma V^T \approx B_{\text{compressed}} \times A_{\text{compressed}}$$
  Using Power Iteration with Hotelling Deflation, we extract the top-$k$ principal singular components. By the **Eckart-Young-Mirsky Theorem**, this produces the mathematically optimal rank-$r$ approximation, preserving $>92\%$ of multi-task variance while keeping parameter count strictly invariant!

### 3. TIES Merging & DARE Sparsification
* **Parameter Interference**: Naively averaging fine-tuned weights causes directional cancellation: positive updates from Task A cancel negative updates from Task B.
* **TIES-Merging** resolves this in three stages:
  1. **Trimming**: Retain only the top $\rho\%$ largest magnitude weights (pruning parameter noise).
  2. **Electing Sign**: Compute the majority sign ($+1$ or $-1$) for each parameter coordinate.
  3. **Disjoint Merging**: Average only weights whose sign agrees with the majority direction.
* **DARE (Drop And REscale)** randomly masks weights with probability $p$ and rescales the remaining weights by $\frac{1}{1-p}$. The expected value $\mathbb{E}[W_{\text{dare}}] = W$ is perfectly preserved while slashing adapter density.

### 4. Group Relative Policy Optimization (GRPO) & Verifiable Rewards
* **Verifiable RL vs Subjective Reward Models**: Unlike standard RLHF which uses human preference models that can be gamed or hallucinate, GRPO for reasoning and agent systems scores responses against deterministic ground truth:
  - $R = -1.0$: Blocked by safety barrier (fork bombs, destructive file deletion).
  - $R = -0.8$: Non-zero exit code (syntax errors, command crashes).
  - $R = +0.5$: Clean execution (valid AST syntax, return code 0).
  - $R = +1.0$: Verified execution passing hermetic unit tests.
* **DPO Contrastive Harvest**: Models that fail the DSCE Oracle gate generate contrastive pairs:
  $$\Delta R = R_{\text{chosen}} - R_{\text{rejected}} > 0$$

### 5. Memory Discipline: Completion-Only Cross-Entropy & Chunked Logits
* **Completion-Only Masking**: Prompts are passed to `input_ids` but masked with $-100$ in `labels`. PyTorch's `cross_entropy` ignores labels equal to $-100$, computing loss and backpropagation exclusively on assistant generation tokens.
* **Chunked Logits**: Computing the full vocabulary projection matrix $B \times L \times V$ across context length $2048$ requires over $1.25\text{ GB}$ of intermediate activation VRAM. Chunked logits evaluates only target prediction tokens, reducing peak activation memory by $75\%$ and enabling 7B parameter models to train comfortably on 8GB consumer GPUs.

### 6. Thermal Duty Cycle & Velocity PI Pacing
* **Hardware Protection**: Consumer GPUs (e.g. RTX 3070 8GB) have compact cooling fans. Sustained 100% duty cycle pushes junction temperatures past 85°C, triggering hardware thermal downclocking.
* **Velocity-Form PI Controller**:
  $$\text{duty}_{t} = \text{duty}_{t-1} - K_p (T_t - T_{t-1}) - K_i (T_t - T_{\text{target}})$$
  Dynamically modulates micro-sleep intervals, maintaining a steady 78.5°C operating plateau with zero hardware throttling.

---

## Directory Layout

```
sovereign_lora_lab/
├── core/
│   ├── sovereign_sft_lora.py      # Completion-only SFT LoRA / QLoRA trainer
│   ├── grpo_reward_scorer.py      # Verifiable GRPO reward scorer & SQLite logger
│   ├── autonomous_dsce_trainer.py # Autonomous DSCE Oracle & DPO harvester
│   ├── adapter_merger.py          # Multi-LoRA SVD & TIES fusion engine
│   └── thermal_monitor.py         # Hardware telemetry & thermal duty cycle pacer
├── recipes/
│   ├── rtx3070_8gb_qlora.yaml     # 8GB VRAM consumer GPU recipe (NF4, paged AdamW)
│   ├── apple_silicon_mps.yaml     # Apple Silicon M-series unified memory recipe
│   └── consumer_gpu_24gb.yaml     # 16-24GB VRAM recipe (BF16 LoRA)
├── datasets/
│   ├── schema.json                # Standard JSON schema for SFT, DPO, and GRPO
│   ├── verify_dataset.py          # Dataset quality gate, deduplication & split tool
│   ├── sample_tooluse_pairs.jsonl # Sample synthetic tool-use training pairs
│   ├── sample_code_repair_pairs.jsonl # Sample DPO code repair preference pairs
│   └── prompt_templates/          # ChatML, Qwen Coder, and OpenAI tool schemas
├── docs/
│   ├── MULTI_LORA_FUSION.md       # SVD rank compression & task arithmetic
│   ├── ADAPTER_MERGING_GUIDE.md   # Architectural foundations of adapter synthesis
│   ├── GGUF_EXPORT_AND_QUANTIZATION.md # HF merge & llama.cpp quantization
│   └── CONSUMER_HARDWARE_GUIDELINES.md # 8GB VRAM budgets & thermal pacing
├── tests/                         # Hermetic standalone unit test suite (33 tests)
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Dependencies
└── LICENSE                        # Dual Apache 2.0 / MIT license
```

---

## Quickstart Tutorial

### 1. Installation
```bash
git clone https://github.com/Ethan-da-Tech-Wizard/sovereign-lora-lab.git
cd sovereign-lora-lab
pip install -e .
```

### 2. Dataset Verification & Quality Gate
Verify dataset structure, remove duplicates, and generate disjoint family-held-out splits:
```bash
python datasets/verify_dataset.py datasets/sample_tooluse_pairs.jsonl --check-ast --split
```

### 3. Dry-Run SFT Pipeline
Validate model configuration, prompt rendering, and dataset tokenization without allocating GPU VRAM:
```bash
python core/sovereign_sft_lora.py \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --data datasets/sample_tooluse_pairs.jsonl \
  --dry-run
```

### 4. Fine-Tuning on 8GB Consumer GPU (RTX 3070 / Apple Silicon)
```bash
python core/sovereign_sft_lora.py \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --data datasets/sample_tooluse_pairs.jsonl \
  --out-root ./runs \
  --run my_adapter_v1 \
  --r 16 \
  --alpha 32 \
  --microbatch 1 \
  --accum 16 \
  --max-len 1024 \
  --lr 2e-4
```

### 5. Multi-LoRA Fusion (Truncated SVD $3r \to r$)
Combine independent skill adapters into a unified master adapter:
```python
from core.adapter_merger import LoRAAdapter, truncated_svd_merge

adapter_tooluse = LoRAAdapter("tools", A_tools, B_tools, alpha=32.0, r=16)
adapter_repair = LoRAAdapter("repair", A_repair, B_repair, alpha=32.0, r=16)

# Merge with custom domain importance weighting
B_fused, A_fused = truncated_svd_merge(
    [adapter_tooluse, adapter_repair],
    target_rank=16,
    weights=[0.60, 0.40]
)
```

### 6. GGUF Export & Quantization
Export adapter conversion command for `llama.cpp`:
```bash
python core/sovereign_sft_lora.py \
  --export-adapter-gguf \
  --adapter-dir ./runs/my_adapter_v1/checkpoint-final \
  --base Qwen/Qwen2.5-Coder-7B-Instruct
```

---

## Hermetic Unit Test Suite

The test suite runs 100% hermetically in standard Python without external cloud services:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
Output:
```
Ran 33 tests in 0.035s
OK
```

---

## Author & Open Source Ecosystem

Created and maintained by [Ethan-da-Tech-Wizard](https://github.com/Ethan-da-Tech-Wizard).

Explore related high-assurance open source security and autonomous systems projects:
* [**RelayScrub**](https://github.com/Ethan-da-Tech-Wizard/relayscrub) — Zero-dependency, offline deterministic privacy scrubber, cryptographic nonce fencer, and prompt injection detector for autonomous AI agent pipelines.
* [**SovereignLoRA Lab**](https://github.com/Ethan-da-Tech-Wizard/sovereign-lora-lab) — Local edge LLM fine-tuning, Direct Preference Optimization (DPO), and Reinforcement Learning (GRPO) engine.
* [**GitHub Profile**](https://github.com/Ethan-da-Tech-Wizard) — Systems programming, deterministic verification, and agentic AI architectures.

---

## License

Dual-licensed under either:
- **Apache License, Version 2.0**
- **MIT License**
