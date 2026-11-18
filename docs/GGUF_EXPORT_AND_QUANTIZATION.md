# GGUF Export & Quantization Workflow

## 1. Overview

Exporting trained LoRA adapters into quantized GGUF format enables sub-10ms edge inference with local inference servers like `llama.cpp`, `ollama`, or custom C++ runtimes.

This guide provides the complete end-to-end pipeline:
1. Merge LoRA adapter into base model weights (in FP16 or BF16).
2. Convert merged model directory to GGUF format.
3. Quantize the GGUF model into `Q4_K_M` (recommended for 8GB VRAM) and `Q8_0` (maximum fidelity).

---

## 2. Option A: Standalone GGUF Adapter Export

If your local server supports runtime dynamic LoRA loading (`--lora path/to/adapter.gguf`):

```bash
python core/sovereign_sft_lora.py \
  --export-adapter-gguf \
  --adapter-dir ./runs/run_sft_lora/checkpoint-final \
  --base Qwen/Qwen2.5-Coder-7B-Instruct \
  --gguf-out ./runs/run_sft_lora/adapter-f16.gguf
```

---

## 3. Option B: Full Base Weight Merge & Quantization (Recommended)

### Step 1: Merge LoRA into Hugging Face Base Model
```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_dir = "./runs/run_sft_lora/checkpoint-final"
output_merged_path = "./models/merged_qwen_7b"

# Load base model in full precision
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Attach and merge adapter
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = model.merge_and_unload()

# Save merged weights
merged_model.save_pretrained(output_merged_path)
tokenizer.save_pretrained(output_merged_path)
print(f"Merged model saved to {output_merged_path}")
```

### Step 2: Convert Merged Model to FP16 GGUF
Using `convert_hf_to_gguf.py` from `llama.cpp`:
```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  ./models/merged_qwen_7b \
  --outtype f16 \
  --outfile ./models/qwen_7b_f16.gguf
```

### Step 3: Quantize to Q4_K_M and Q8_0
Using `llama-quantize` binary:
```bash
# High-Efficiency Edge Quantization (4-bit K-quant Medium)
llama-quantize ./models/qwen_7b_f16.gguf ./models/qwen_7b_q4_k_m.gguf Q4_K_M

# Studio High-Precision Quantization (8-bit)
llama-quantize ./models/qwen_7b_f16.gguf ./models/qwen_7b_q8_0.gguf Q8_0
```

---

## 4. Serving on Edge Hardware

To serve the quantized model locally via `llama-server`:
```bash
llama-server \
  --model ./models/qwen_7b_q4_k_m.gguf \
  --port 18190 \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --threads 8
```
This serves an OpenAI-compatible endpoint at `http://127.0.0.1:18190/v1/chat/completions`.
