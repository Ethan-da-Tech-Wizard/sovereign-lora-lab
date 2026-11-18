# Consumer GPU Hardware Guidelines

## 1. Hardware Sizing & Memory Budgeting

Running LoRA/QLoRA fine-tuning on consumer hardware (such as 8GB VRAM NVIDIA RTX 3070 or Apple Silicon M-series laptops) requires strict discipline over VRAM allocation.

The total VRAM consumption during training decomposes into:
$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{adapters}} + \text{VRAM}_{\text{activations}} + \text{VRAM}_{\text{optimizer}} + \text{CUDA Overhead}$$

### Memory Budget Breakdown (7B Model on 8GB VRAM)
| Component | Precision / Configuration | VRAM Footprint |
|---|---|---|
| **Base Model Weights** | NF4 Quantized (BitsAndBytes 4-bit) | ~3.80 GB |
| **LoRA Adapters** | $r=16, \alpha=32$ (All linear projections) | ~0.15 GB |
| **Optimizer States** | Paged 8-bit AdamW (`paged_adamw_8bit`) | ~0.35 GB |
| **Activations** | Microbatch=1, SeqLen=1024, Chunked Logits | ~0.45 GB |
| **CUDA / Driver Overhead** | Context, CUDA runtime buffer | ~1.20 GB |
| **Total Peak Footprint** | | **~5.95 GB** (Safely within 8GB limit) |

> [!WARNING]
> Without **Chunked Logits** and **Gradient Checkpointing**, intermediate forward activations alone can spike above 4.5 GB, causing immediate Out-Of-Memory (OOM) errors on 8GB GPUs. SovereignLoRA enforces chunked cross-entropy and gradient checkpointing by default.

---

## 2. NVIDIA RTX 3070 (8GB VRAM) Operational Rules

1. **Thermal Duty Cycle:** Mobile and compact desktop RTX 3070 GPUs experience steep thermal ramps under sustained matrix multiply. The built-in `ThermalPacer` monitors GPU temperature at step boundaries:
   - `<80°C`: Full speed.
   - `80°C - 83°C`: Soft duty cycle sleep (2s - 6s sleep per step).
   - `84°C - 86°C`: Emergency pause; halts optimizer steps until the die cools to `<=78°C`.
   - `>=90°C`: Immediate fail-closed safety abort and checkpoint save.
2. **Strict 96°C CPU Safety Ceiling:** Continuous CPU monitoring ensures system stability; training halts if CPU package reaches 96°C.
3. **Microbatch Sizing:** Keep `--microbatch 1` and accumulate gradients (`--accum 16`) to reach an effective batch size of 16.

---

## 3. Apple Silicon M-Series (Unified Memory / MPS)

1. **Unified Memory Sharing:** Apple Silicon shares system RAM dynamically between CPU and GPU cores. On a 16GB Mac, allocate no more than 10GB to PyTorch MPS buffers.
2. **Float16 Compute:** Apple Neural Engine and GPU cores excel with `torch.float16` or `torch.bfloat16`.
3. **Memory Pressure Detection:** Use macOS memory pressure telemetry. If memory pressure shifts to Level 2 (Warning) or Level 4 (Critical), `ThermalSafetyGuard` automatically throttles execution to prevent swapping or OS freeze.
