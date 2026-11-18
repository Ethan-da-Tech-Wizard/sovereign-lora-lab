# SovereignLoRA

**Local Edge LLM Fine-Tuning, Direct Preference Optimization (DPO), and Reinforcement Learning (GRPO) Engine.**

SovereignLoRA is an open-source, production-ready framework for training, aligning, and deploying state-of-the-art language models on consumer hardware (such as 8GB VRAM NVIDIA RTX 3070 and Apple Silicon M-series).

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

## Key Features

1. **Memory-Disciplined SFT Trainer (`core/sovereign_sft_lora.py`)**:
   - **Completion-Only Cross-Entropy**: Computes loss and backpropagation exclusively on assistant completion tokens, avoiding prompt gradient waste.
   - **Chunked Logits**: Prevents intermediate activation explosions, fitting 7B parameter models within 6GB usable VRAM.
   - **Exact Prefix Sharing (Tree Packing)**: Packs multiple completion variants with identical prompt prefixes into unified causal trees, cutting token processing overhead by up to 40%.

2. **Continuous Thermal Duty Cycle (`core/thermal_monitor.py`)**:
   - Velocity-form PI duty-cycle software controller dynamically modulates training speed.
   - Prevents thermal throttling: Soft pacing at 80°C, pause at 84°C, fail-closed safety abort at 90°C GPU / 96°C CPU.

3. **Verifiable GRPO Reward Scorer (`core/grpo_reward_scorer.py`)**:
   - Computes deterministic mathematical rewards $R \in [-1.0, +1.0]$.
   - Hermetic AST syntax validation, command execution safety gating, and test runner verification.
   - Evaluates contrastive preference margins: $\Delta R = R_{\text{chosen}} - R_{\text{rejected}}$.

4. **Deterministic Semantic Consistency Engine (DSCE) Trainer (`core/autonomous_dsce_trainer.py`)**:
   - Symbolic knowledge vials prove ground-truth answers before promoting candidate model responses.
   - Automatically harvests contrastive DPO preference pairs from model failures.

5. **Multi-LoRA Adapter Fusion (`core/adapter_merger.py`)**:
   - Pure Python standard library implementation (with optional NumPy acceleration).
   - **Linear Task Arithmetic**: Fast weighted combination.
   - **Truncated SVD Rank Compression**: Compresses $3r \to r$ (e.g. rank 48 back to canonical rank 16), preserving $>92\%$ of multi-task variance with zero parameter expansion.
   - **TIES-Merging & DARE**: Resolves parameter sign collisions and applies drop-and-rescale sparsification.

6. **GGUF Export & Quantization Pipeline**:
   - Integrates with `llama.cpp` tools to produce `Q4_K_M` (edge latency) and `Q8_0` (maximum fidelity) quantized weights.

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
git clone https://github.com/your-org/sovereign_lora_lab.git
cd sovereign_lora_lab
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
Export adapter conversion command:
```bash
python core/sovereign_sft_lora.py \
  --export-adapter-gguf \
  --adapter-dir ./runs/my_adapter_v1/checkpoint-final \
  --base Qwen/Qwen2.5-Coder-7B-Instruct
```

---

## Unit Test Suite

The test suite runs 100% hermetically in standard Python without external cloud services:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
Output:
```
Ran 33 tests in 0.032s
OK
```

---

## License

Dual-licensed under either:
- **Apache License, Version 2.0**
- **MIT License**
