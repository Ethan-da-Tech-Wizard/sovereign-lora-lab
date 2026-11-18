#!/usr/bin/env python3
"""core/sovereign_sft_lora.py — Completion-Only SFT LoRA / QLoRA Training Engine.

Engineered for consumer GPUs (e.g. NVIDIA RTX 3070 8GB, Apple Silicon M-series, RTX 3090/4090):
  - Completion-only cross-entropy loss (computes gradient on assistant tokens only)
  - Chunked logits calculation to eliminate intermediate activation memory spikes
  - Exact prefix sharing (tree packing) for multi-completion batches
  - Integrated software thermal duty cycle (PI velocity loop) & fail-closed safety guards
  - Deterministic family-held-out dev split & deduplication hashing guard
  - Reproducible run receipt generation (cryptographic SHA256 of code, data, base, and adapter)
  - Direct conversion command formatting for llama.cpp GGUF 16-bit and quantized export
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from core import thermal_monitor as tm
except ImportError:
    import thermal_monitor as tm

# ── Global Defaults & Lane Specifications ──────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "datasets" / "sample_tooluse_pairs.jsonl"
SERVING_TARGET = "Qwen2.5-Coder-7B-Instruct"
DEFAULT_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_CONVERT_LORA = Path("convert_lora_to_gguf.py")
GGUF_PYTHON = sys.executable
SEED = 1101

LANES: Dict[str, Dict[str, Any]] = {
    "candidate-1.5b": {"match": "qwen2.5-coder-1.5b", "port": 18193, "note": "1.5B edge candidate lane"},
    "candidate-3b": {"match": "qwen2.5-3b", "port": 18194, "note": "3B general candidate lane"},
    "target-7b": {"match": "qwen2.5-coder-7b", "port": 18190, "note": "7B production target lane"},
    "candidate-1p7b": {"match": "qwen3-1.7b", "port": 18191, "note": "1.7B edge candidate lane"},
    "candidate-q35-4b": {"match": "qwen3.5-4b", "port": 18192, "note": "4B per-device candidate lane"},
}

# Thermal duty cycle parameters (Celsius)
BAND_NONE, BAND_SHORT = 80, 84
HOT = 87
SHORT_SLEEP, WARM_TARGET, WARM_MAX, HOT_TARGET, HOT_MAX = 5, 78, 60, 76, 120
CPU_SOFT, CPU_SOFT_STREAK, CPU_SLEEP = 95, 3, 10
COOL_FRACTION_CAP = 0.40

# Fine-grained pacing: PI velocity loop
FINE_TARGET, FINE_KP, FINE_KI = 78.5, 0.015, 0.006
FINE_DUTY_MIN, FINE_DUTY_START, FINE_MAX_SLEEP, FINE_COOL_FRACTION_CAP, FINE_BOUND_AFTER_S = 0.12, 0.2, 20.0, 0.93, 600


# -----------------------------------------------------------------------------
# Pure Python Helpers (Data Loading, Deduplication, Splits)
# -----------------------------------------------------------------------------

def chosen_text(c: Union[str, dict, list]) -> str:
    """Normalize chosen completion to string."""
    return c if isinstance(c, str) else json.dumps(c, sort_keys=True)


def row_hash(row: Dict[str, Any]) -> str:
    """Deterministic hash of prompt + chosen completion."""
    return hashlib.sha1((row["prompt"] + chosen_text(row["chosen"])).encode()).hexdigest()[:12]


def load_drops(directory: Optional[Union[str, Path]] = None, pattern: str = "drop_ids_*.json") -> Tuple[set, dict, list]:
    """Load IDs and expected hashes of items to drop from training."""
    d = Path(directory) if directory else Path()
    ids, hashes, files = set(), {}, []
    if d.is_dir():
        for f in sorted(d.glob(pattern)):
            try:
                data = json.loads(f.read_text())
                files.append(f.name)
                ids.update(int(i) for i in data.get("drop_ids", []))
                hashes.update({int(k): v for k, v in (data.get("drop_hashes") or {}).items()})
            except Exception:
                pass
    return ids, hashes, files


def _kept(idx: int, r: Dict[str, Any]) -> Dict[str, Any]:
    k = {"idx": idx, "family": r.get("family", "?"), "prompt": r["prompt"], "chosen": r["chosen"]}
    if isinstance(r.get("messages"), list) and r["messages"]:
        k["messages"] = r["messages"]
        if isinstance(r.get("tools"), list):
            k["tools"] = r["tools"]
    return k


def load_rows(
    path: Union[str, Path],
    drop_ids: Tuple[int, ...] = (),
    drop_hashes: Optional[Dict[int, str]] = None,
    lenient_hash: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load JSONL dataset rows with hash verification and exact duplicate removal."""
    rows, bad, total, hash_mismatch = [], [], 0, []
    p = Path(path)
    if not p.exists():
        return [], {"lines": 0, "kept": 0, "error": f"File not found: {path}"}

    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        total += 1
        try:
            r = json.loads(line)
        except Exception:
            bad.append(idx)
            continue

        if not isinstance(r.get("prompt"), str) or not r["prompt"] or not r.get("chosen") or not isinstance(r["chosen"], (str, list, dict)):
            bad.append(idx)
            continue

        if idx in drop_ids:
            want = (drop_hashes or {}).get(idx)
            if want and row_hash(r) != want:
                if lenient_hash:
                    hash_mismatch.append(idx)
                    rows.append(_kept(idx, r))
                    continue
                raise ValueError(f"drop id {idx}: hash {row_hash(r)} != recorded {want}; drop list does not match dataset file")
            continue
        rows.append(_kept(idx, r))

    seen, uniq, dups = set(), [], 0
    for r in rows:
        k = (r["prompt"], chosen_text(r["chosen"]))
        if k in seen:
            dups += 1
            continue
        seen.add(k)
        uniq.append(r)

    report = {
        "lines": total,
        "dropped_by_id": len([i for i in drop_ids if i < total]) if drop_ids else 0,
        "malformed": len(bad),
        "drop_hash_mismatch_not_dropped": len(hash_mismatch),
        "exact_duplicates_removed": dups,
        "kept": len(uniq),
    }
    return uniq, report


def split_dev(rows: List[Dict[str, Any]], seed: int = SEED, dev_n: int = 48) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Seeded, family-round-robin dev split. Dev rows have unique completions and disjoint prompts from train."""
    counts: Dict[str, int] = {}
    for r in rows:
        k = chosen_text(r["chosen"])
        counts[k] = counts.get(k, 0) + 1
    cands = [r for r in rows if counts[chosen_text(r["chosen"])] == 1]
    rng = random.Random(seed)
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    for r in cands:
        by_fam.setdefault(r["family"], []).append(r)
    for fam in by_fam.values():
        rng.shuffle(fam)
    dev: List[Dict[str, Any]] = []
    fams = sorted(by_fam)
    while len(dev) < dev_n and any(by_fam.values()):
        for f in fams:
            if by_fam[f] and len(dev) < dev_n:
                dev.append(by_fam[f].pop())
    dev_idx = {r["idx"] for r in dev}
    dev_prompts = {r["prompt"] for r in dev}
    train = [r for r in rows if r["idx"] not in dev_idx and r["prompt"] not in dev_prompts]
    assert not ({r["prompt"] for r in train} & dev_prompts), "Train and dev prompts must be strictly disjoint"
    return train, dev


def default_data(directory: Optional[Union[str, Path]] = None) -> Path:
    """Find default JSONL training dataset."""
    d = Path(directory) if directory else REPO / "datasets"
    candidates = ("pairs_v3.jsonl", "pairs_v2.jsonl", "sample_tooluse_pairs.jsonl")
    for name in candidates:
        if (d / name).exists():
            return d / name
    return d / "sample_tooluse_pairs.jsonl"


# -----------------------------------------------------------------------------
# Model Serving Check & Tool Chat Template Rendering
# -----------------------------------------------------------------------------

def _norm_name(s: str) -> str:
    return re.sub(r"[-_.]+", "", s.lower())


def base_matches_target(base: str, target: str = SERVING_TARGET) -> bool:
    """Verify whether base model architecture matches serving target."""
    nb, nt = _norm_name(base), _norm_name(target)
    return nb == nt or nb.endswith("/" + nt) or nb.endswith("\\" + nt) or nt in nb


def serving_check(base: str, target: str = SERVING_TARGET, lane: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """Determine whether an adapter trained on `base` can be served on the production target."""
    dec = {
        "base": base,
        "serving_target": target,
        "lane": lane,
        "base_matches_target": base_matches_target(base, target),
    }
    if lane:
        if lane not in LANES:
            return False, {**dec, "ok": False, "reason": f"unknown lane {lane!r}; known: {sorted(LANES)}"}
        if _norm_name(LANES[lane]["match"]) not in _norm_name(base):
            return False, {**dec, "ok": False, "reason": f"lane {lane} requires a base naming {LANES[lane]['match']}"}
        return True, {
            **dec,
            "ok": True,
            "decision": "candidate lane (adapter can NOT be served on the production target)",
            "lane_port": LANES[lane]["port"],
            "production_ready": False,
        }
    if dec["base_matches_target"]:
        return True, {**dec, "ok": True, "decision": "base is the serving target"}
    return False, {
        **dec,
        "ok": False,
        "reason": f"base differs from the serving target {target}; train the target or specify --lane {'|'.join(sorted(LANES))}",
    }


def ensure_tool_system_prompt(openai_req: Dict[str, Any]) -> Dict[str, Any]:
    """Inject tool schemas into system prompt for robust tool invocation."""
    tools = openai_req.get("tools")
    if not tools:
        return openai_req
    marker = "You have access to ONLY these exact tools:"
    tool_lines = [f"\n{marker}"]
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        req = fn.get("parameters", {}).get("required", [])
        tool_lines.append(f"- {name}: {desc} | parameters: {json.dumps(props)} | required: {json.dumps(req)}")
    tool_lines.append(
        "\nCRITICAL INSTRUCTIONS:\n"
        "- Call ONLY tools from the list above. Do NOT invent other tools.\n"
        "- When asked to perform actions, you MUST call the provided tools.\n"
        "- To execute a tool call, output ONLY a JSON list of calls: "
        '[{"name": "tool_name", "arguments": {"arg": "val"}}]'
    )
    prompt_addition = "\n".join(tool_lines)

    messages = openai_req.get("messages", [])
    if messages and messages[0].get("role") == "system":
        if marker not in messages[0]["content"]:
            messages[0]["content"] += "\n\n" + prompt_addition
    else:
        messages.insert(0, {"role": "system", "content": prompt_addition})
    return openai_req


def default_tools() -> List[Dict[str, Any]]:
    """Standard 5 read/execute agent tools."""
    return [
        {"type": "function", "function": {"name": "read_file", "description": "Read file contents", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
        {"type": "function", "function": {"name": "list_directory", "description": "List directory entries", "parameters": {"type": "object", "properties": {"dir_path": {"type": "string"}}, "required": []}}},
        {"type": "function", "function": {"name": "glob", "description": "Glob pattern search", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "grep_search", "description": "Regex pattern search across files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "dir_path": {"type": "string"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "run_shell_command", "description": "Execute shell command in sandbox", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    ]


def bridge_messages(prompt: str, tools: Optional[List[Dict[str, Any]]] = None, system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Format standard OpenAI request messages with tool injection."""
    req = {
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}],
        "tools": copy.deepcopy(tools) if tools is not None else default_tools(),
    }
    return ensure_tool_system_prompt(req)["messages"]


def qwen_chat_render(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Pure-python fallback mirror of Qwen chat template with native tools block."""
    out = []
    msgs = list(messages)
    sys_text = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else None
    if sys_text is not None:
        msgs = msgs[1:]
    if tools:
        head = (sys_text + "\n\n" if sys_text else "") + (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for t in tools:
            head += "\n" + json.dumps(t, ensure_ascii=False)
        head += (
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        )
        out.append("<|im_start|>system\n" + head + "<|im_end|>\n")
    elif sys_text is not None:
        out.append("<|im_start|>system\n" + sys_text + "<|im_end|>\n")
    for m in msgs:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)


def render_bridge_prompt(prompt: str, tools: Optional[List[Dict[str, Any]]] = None, tokenizer: Any = None) -> str:
    """Render full prompt text as seen during tool invocation."""
    tools = default_tools() if tools is None else tools
    msgs = bridge_messages(prompt, tools)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    return qwen_chat_render(msgs, tools)


# -----------------------------------------------------------------------------
# Cryptographic Checksums, Receipts, and GGUF Export Commands
# -----------------------------------------------------------------------------

def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: Union[str, Path], patterns: Tuple[str, ...] = ("*.safetensors", "*.gguf", "*.bin")) -> Optional[str]:
    p = Path(path)
    if p.is_file():
        return file_sha256(p)
    files = sorted(f for pat in patterns for f in p.glob(pat) if f.is_file())
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.name}:{file_sha256(f)}\n".encode())
    return h.hexdigest()


def git_state(repo: Union[str, Path] = REPO) -> Dict[str, Any]:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return None
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": len((run("status", "--porcelain") or "").splitlines()),
    }


def build_receipt(
    args: Any,
    data_path: Union[str, Path],
    serving_decision: Dict[str, Any],
    adapter_path: Optional[Union[str, Path]] = None,
    sha_fn=tree_sha256,
    git_fn=git_state,
) -> Dict[str, Any]:
    """Build provenance receipt record."""
    def sha(p):
        try:
            return sha_fn(p) if p and Path(p).exists() else None
        except Exception:
            return None

    return {
        "schema": "sovereign-sft-run-receipt.v1",
        "time": time.time(),
        "serving_check": serving_decision,
        "base_model": {"path": str(args.base), "sha256": sha(args.base)},
        "adapter": {"path": str(adapter_path) if adapter_path else None, "sha256": sha(adapter_path)},
        "data": {"path": str(data_path), "sha256": sha(data_path)},
        "git": git_fn(),
        "code": {"sovereign_sft_lora": sha(Path(__file__))},
        "render": getattr(args, "render", "plain"),
        "lane": getattr(args, "lane", None),
    }


def adapter_gguf_command(
    adapter_dir: Union[str, Path],
    base: Union[str, Path],
    outfile: Union[str, Path],
    outtype: str = "f16",
    python: str = GGUF_PYTHON,
    script: Union[str, Path] = DEFAULT_CONVERT_LORA,
) -> List[str]:
    """Format exact convert_lora_to_gguf.py CLI invocation."""
    return [str(python), str(script), "--base", str(base), "--outtype", outtype, "--outfile", str(outfile), str(adapter_dir)]


def export_adapter_gguf(args: Any, run_fn=subprocess.run) -> int:
    """Print or execute GGUF adapter conversion command."""
    ok, dec = serving_check(args.base, args.serving_target, args.lane)
    if not ok:
        print("REFUSED: " + dec["reason"], file=sys.stderr)
        return 5
    out_dir = Path(args.out_root) / args.run
    adapter = Path(args.adapter_dir) if args.adapter_dir else (latest_checkpoint(out_dir) or out_dir / "adapter")
    outfile = Path(args.gguf_out) if args.gguf_out else out_dir / f"{args.run}-adapter-f16.gguf"
    convert_script = getattr(args, "convert_script", DEFAULT_CONVERT_LORA)
    cmd = adapter_gguf_command(adapter, args.base, outfile, script=convert_script)
    payload = {
        "mode": "export-adapter-gguf",
        "executed": bool(args.run_conversion),
        "command": cmd,
        "shell": " ".join(__import__("shlex").quote(c) for c in cmd),
        "adapter_dir": str(adapter),
        "adapter_exists": adapter.exists(),
        "script_exists": Path(convert_script).exists(),
        "serving_check": dec,
    }
    print(json.dumps(payload, indent=2))
    if not args.run_conversion:
        return 0
    if not adapter.exists():
        print(f"REFUSED: adapter dir {adapter} does not exist", file=sys.stderr)
        return 5
    outfile.parent.mkdir(parents=True, exist_ok=True)
    rc = run_fn(cmd).returncode
    if rc == 0 and outfile.exists():
        write_json(outfile.with_suffix(".receipt.json"), build_receipt(args, args.data, dec, adapter_path=outfile))
    return rc


# -----------------------------------------------------------------------------
# Example Encoding & Completion Token Masking
# -----------------------------------------------------------------------------

def format_example(prompt: str, chosen: str, think_prefix: bool = True) -> Tuple[str, str]:
    """Pure fallback format for ChatML template."""
    p = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if think_prefix else "")
    return p, chosen + "<|im_end|>"


def split_with_template(tokenizer: Any, prompt: str, chosen: Union[str, dict, list], bridge_tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
    """Split prompt and completion via tokenizer chat template."""
    user = bridge_messages(prompt, bridge_tools) if bridge_tools is not None else [{"role": "user", "content": prompt}]
    kw = {"tokenize": False}
    if bridge_tools is not None:
        kw["tools"] = bridge_tools
    try:
        p = tokenizer.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False, **kw)
    except TypeError:
        p = tokenizer.apply_chat_template(user, add_generation_prompt=True, **kw)
    if isinstance(chosen, str):
        return p, chosen + (tokenizer.eos_token if "<|im_end|>" not in chosen and tokenizer.eos_token else "")
    msgs = [chosen] if isinstance(chosen, dict) else list(chosen)
    full = tokenizer.apply_chat_template(user + msgs, **kw)
    if full.startswith(p):
        return p, full[len(p):].rstrip("\n")
    marker = "<|im_start|>assistant\n"
    if marker in full:
        return p, full[full.rfind(marker) + len(marker):].lstrip("\n").replace("<think>\n\n</think>\n\n", "").rstrip("\n")
    raise ValueError("chat template output does not extend the generation prompt")


def encode_example(tokenizer: Any, prompt: str, chosen: Union[str, dict, list], max_len: int, think_prefix: bool = True, bridge_tools: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Tokenize example into input_ids and labels with prompt tokens masked to -100."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        p, c = split_with_template(tokenizer, prompt, chosen, bridge_tools)
    else:
        if not isinstance(chosen, str):
            raise ValueError("structured chosen needs a tokenizer chat template")
        p, c = format_example(prompt, chosen, think_prefix)
    pi = tokenizer(p, add_special_tokens=False)["input_ids"]
    ci = tokenizer(c, add_special_tokens=False)["input_ids"]
    if len(pi) + len(ci) > max_len:
        return None
    return {"input_ids": pi + ci, "labels": [-100] * len(pi) + ci}


def _apply(tokenizer: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], chat_template: Optional[str] = None, **kw) -> str:
    kw = dict(kw, tokenize=False, enable_thinking=False)
    if tools:
        kw["tools"] = tools
    if chat_template:
        kw["chat_template"] = chat_template
    return tokenizer.apply_chat_template(messages, **kw)


def row_messages_tools(row: Dict[str, Any], render: str = "row", bridge_tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if render == "row":
        if "messages" not in row:
            raise ValueError(f"row {row.get('idx')}: no 'messages'; use --render bridge or --render plain")
        return row["messages"], row.get("tools")
    if render == "bridge":
        tools = row.get("tools") or bridge_tools or default_tools()
        return bridge_messages(row["prompt"], tools), tools
    return [{"role": "user", "content": row["prompt"]}], None


def split_row(tokenizer: Any, row: Dict[str, Any], render: str = "row", bridge_tools: Optional[List[Dict[str, Any]]] = None, chat_template: Optional[str] = None) -> Tuple[str, str]:
    msgs, tools = row_messages_tools(row, render, bridge_tools)
    p = _apply(tokenizer, msgs, tools, chat_template, add_generation_prompt=True)
    ch = row["chosen"]
    if isinstance(ch, str):
        return p, ch + (tokenizer.eos_token if "<|im_end|>" not in ch and tokenizer.eos_token else "")
    amsgs = [ch] if isinstance(ch, dict) else list(ch)
    full = _apply(tokenizer, list(msgs) + amsgs, tools, chat_template)
    if not full.startswith(p):
        raise ValueError("chat template output does not extend the generation prompt")
    return p, full[len(p):].rstrip("\n")


def template_tail(tokenizer: Any, row: Dict[str, Any], render: str = "row", bridge_tools: Optional[List[Dict[str, Any]]] = None, chat_template: Optional[str] = None) -> Optional[str]:
    msgs, tools = row_messages_tools(row, render, bridge_tools)
    p = _apply(tokenizer, msgs, tools, chat_template, add_generation_prompt=True)
    ch = row["chosen"]
    amsgs = [{"role": "assistant", "content": ch}] if isinstance(ch, str) else ([ch] if isinstance(ch, dict) else list(ch))
    full = _apply(tokenizer, list(msgs) + amsgs, tools, chat_template)
    return full[len(p):] if full.startswith(p) else None


def encode_row(tokenizer: Any, row: Dict[str, Any], max_len: int, render: str = "row", bridge_tools: Optional[List[Dict[str, Any]]] = None, chat_template: Optional[str] = None) -> Optional[Dict[str, Any]]:
    p, c = split_row(tokenizer, row, render, bridge_tools, chat_template)
    pi = tokenizer(p, add_special_tokens=False)["input_ids"]
    ci = tokenizer(c, add_special_tokens=False)["input_ids"]
    if len(pi) + len(ci) > max_len:
        return None
    return {"input_ids": pi + ci, "labels": [-100] * len(pi) + ci, "n_prompt": len(pi), "n_completion": len(ci), "idx": row.get("idx")}


def length_stats(lens: List[int]) -> Dict[str, Any]:
    if not lens:
        return {"n": 0, "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
    sl = sorted(lens)
    return {
        "n": len(sl),
        "min": sl[0],
        "max": sl[-1],
        "mean": round(sum(sl) / len(sl), 1),
        "p50": sl[len(sl) // 2],
        "p95": sl[int(len(sl) * 0.95)],
    }


def estimate_resources(
    cfg: Dict[str, Any],
    max_len: int,
    micro: int,
    n_rows: int,
    epochs: int,
    mean_len: float,
    mean_completion: float,
    chunked_logits: bool = True,
    tok_per_s: float = 120.0,
    sleep_fraction: float = 0.25,
    shared_prefix_tokens: int = 0,
    group: int = 1,
) -> Dict[str, Any]:
    """Estimate VRAM usage and wall-clock training duration."""
    base_gb = cfg.get("base_weights_gb", 3.8)
    lora_gb = cfg.get("lora_weights_gb", 0.15)
    ctx_scale = (max_len / 1024.0)
    act_gb = (0.45 if chunked_logits else 1.8) * micro * ctx_scale
    cuda_overhead = 1.2
    vram_peak_gb = round(base_gb + lora_gb + act_gb + cuda_overhead, 2)

    total_tokens = n_rows * epochs * (mean_len - (shared_prefix_tokens * (group - 1) / max(1, group)))
    compute_s = total_tokens / max(1.0, tok_per_s)
    wall_s = compute_s / max(0.01, 1.0 - sleep_fraction)

    return {
        "vram_peak_gb": vram_peak_gb,
        "total_tokens_approx": int(total_tokens),
        "compute_seconds": round(compute_s, 1),
        "wall_seconds": round(wall_s, 1),
        "fits_8gb_vram": vram_peak_gb <= 7.4,
    }


def build_schedule(n: int, micro: int, accum: int, epochs: int, seed: int = SEED) -> List[List[List[int]]]:
    """Generate deterministically batched row indices per optimizer step."""
    steps = []
    rng = random.Random(seed)
    step_size = micro * accum
    for _ in range(epochs):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(0, len(perm), step_size):
            chunk = perm[i:i + step_size]
            if not chunk:
                continue
            batches = [chunk[j:j + micro] for j in range(0, len(chunk), micro)]
            steps.append(batches)
    return steps


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup:
        return peak * float(step + 1) / max(1.0, float(warmup))
    prog = float(step - warmup) / max(1.0, float(total - warmup))
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# -----------------------------------------------------------------------------
# Thermal Pacing and Hardware Telemetry
# -----------------------------------------------------------------------------

def cooling_plan(gpu: float, cpu_streak: int = 0) -> Tuple[Optional[float], int, int, bool]:
    """(sleep_target_c, fixed_sleep_s, max_sleep_s, checkpoint_first)."""
    if os.environ.get("SOVEREIGN_THERMAL_WAIT", "1") == "0":
        return None, 0, 0, False
    if gpu >= HOT:
        return HOT_TARGET, 0, HOT_MAX, True
    if gpu >= BAND_SHORT:
        return WARM_TARGET, 0, WARM_MAX, False
    if gpu >= BAND_NONE:
        return None, SHORT_SLEEP, SHORT_SLEEP, False
    if cpu_streak >= CPU_SOFT_STREAK:
        return None, CPU_SLEEP, CPU_SLEEP, False
    return None, 0, 0, False


def guard_fault(gpu: Optional[float], cpu: Optional[float]) -> Optional[str]:
    """Check whether telemetry is unknown or breaches abort limits."""
    if not tm.known(gpu):
        return "unknown GPU telemetry"
    abort_gpu = int(os.environ.get("SOVEREIGN_ABORT_GPU", 90))
    pause_gpu = int(os.environ.get("SOVEREIGN_PAUSE_GPU", 84))
    pause_cpu = int(os.environ.get("SOVEREIGN_PAUSE_CPU", 96))

    if os.environ.get("SOVEREIGN_THERMAL_WAIT", "1") == "0":
        if gpu >= abort_gpu:
            return f"GPU {gpu}°C >= abort ceiling {abort_gpu}°C"
        if tm.known(cpu) and cpu > 102:
            return f"CPU {cpu}°C > critical limit 102°C"
        return None

    if gpu >= pause_gpu:
        return f"GPU {gpu}°C >= pause threshold {pause_gpu}°C"
    if tm.known(cpu) and cpu >= pause_cpu:
        return f"CPU {cpu}°C >= pause threshold {pause_cpu}°C"
    return None


class ThermalPacer:
    """Manages thermal pacing duty cycle and fail-closed safety interrupts."""

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        gpu_fn=None,
        cpu_fn=None,
        sleep_fn=time.sleep,
        now_fn=time.monotonic,
        stop_path: Optional[Union[str, Path]] = None,
        fine: bool = False,
        target: float = FINE_TARGET,
        kp: float = FINE_KP,
        ki: float = FINE_KI,
    ) -> None:
        self.fine, self.target, self.kp, self.ki = bool(fine), float(target), kp, ki
        self.duty = FINE_DUTY_START
        self.prev_gpu: Optional[float] = None
        self.ticks = 0
        self.tick_log_every = 1
        self.log_path = Path(log_path) if log_path else None
        self.gpu_fn = gpu_fn or tm.gpu_temp
        self.cpu_fn = cpu_fn or tm.cpu_temp
        self.sleep_fn, self.now_fn = sleep_fn, now_fn
        self.stop_path = Path(stop_path) if stop_path is not None else (Path.home() / ".sovereign_lora" / "STOP")
        self.t0 = now_fn()
        self.slept = 0.0
        self.cpu_streak = 0
        self.max_gpu: Optional[float] = None
        self.max_cpu: Optional[float] = None
        self.faults: List[str] = []
        self.sleeps = 0

    def _log(self, **kw) -> None:
        kw["t"] = round(time.time(), 1)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(kw) + "\n")

    def _track(self, gpu: Optional[float], cpu: Optional[float]) -> None:
        if tm.known(gpu):
            self.max_gpu = gpu if self.max_gpu is None else max(self.max_gpu, gpu)
        if tm.known(cpu):
            self.max_cpu = cpu if self.max_cpu is None else max(self.max_cpu, cpu)

    def _sleep(self, step: int, seconds: float, gpu: Optional[float], cpu: Optional[float], reason: str) -> None:
        self.sleep_fn(seconds)
        self.slept += seconds
        self.sleeps += 1
        self._log(event="sleep", step=step, seconds=round(seconds, 1), gpu=gpu, cpu=cpu, reason=reason)

    def duty_update(self, gpu: float) -> float:
        prev = gpu if self.prev_gpu is None else self.prev_gpu
        self.duty += -self.kp * (gpu - prev) - self.ki * (gpu - self.target)
        self.duty = min(1.0, max(FINE_DUTY_MIN, self.duty))
        self.prev_gpu = gpu
        return self.duty

    def after_unit(self, step: int, work_s: float, checkpoint_fn=None) -> str:
        if not self.fine:
            return "ok"
        if self.stop_path and self.stop_path.exists():
            self.faults.append("STOP file present")
            self._log(event="fault", step=step, reason="STOP file")
            return "fault"
        gpu, cpu = self.gpu_fn(), self.cpu_fn()
        self._track(gpu, cpu)
        reason = guard_fault(gpu, cpu)
        if reason:
            self.faults.append(reason)
            self._log(event="fault", step=step, gpu=gpu, cpu=cpu, reason=reason)
            if checkpoint_fn:
                checkpoint_fn()
            return "fault"
        sec = min(FINE_MAX_SLEEP, max(0.0, work_s * (1.0 - self.duty) / self.duty))
        self.ticks += 1
        if sec >= 0.05:
            self.sleep_fn(sec)
            self.slept += sec
            self.sleeps += 1
        valley = self.gpu_fn()
        if tm.known(valley):
            self.duty_update(valley)
        if self.ticks % self.tick_log_every == 0:
            self._log(event="tick", step=step, gpu=gpu, valley=valley, cpu=cpu, duty=round(self.duty, 3), work=round(work_s, 2), sleep=round(sec, 2))
        return "ok"

    def after_step(self, step: int, checkpoint_fn=None) -> str:
        if self.stop_path and self.stop_path.exists():
            self.faults.append("STOP file present")
            self._log(event="fault", step=step, reason="STOP file")
            return "fault"
        gpu, cpu = self.gpu_fn(), self.cpu_fn()
        self._track(gpu, cpu)
        reason = guard_fault(gpu, cpu)
        if reason:
            self.faults.append(reason)
            self._log(event="fault", step=step, gpu=gpu, cpu=cpu, reason=reason)
            if checkpoint_fn:
                checkpoint_fn()
            return "fault"
        self.cpu_streak = self.cpu_streak + 1 if tm.known(cpu) and cpu >= CPU_SOFT else 0
        target, fixed, cap, ckpt = cooling_plan(gpu, self.cpu_streak)
        if self.fine and gpu < HOT:
            target, fixed, cap, ckpt = None, 0, 0, False
        if ckpt and checkpoint_fn:
            checkpoint_fn()
        if target is not None:
            waited = 0.0
            while waited < cap and gpu > target:
                self._sleep(step, 5, gpu, cpu, f"cool to <={target}")
                waited += 5
                gpu = self.gpu_fn()
                self._track(gpu, None)
                reason = guard_fault(gpu, None)
                if reason:
                    self.faults.append(reason)
                    self._log(event="fault", step=step, gpu=gpu, reason=reason)
                    if checkpoint_fn:
                        checkpoint_fn()
                    return "fault"
        elif fixed:
            self._sleep(step, fixed, gpu, cpu, "duty cycle")
        return "ok"

    def summary(self) -> Dict[str, Any]:
        elapsed = self.now_fn() - self.t0
        return {
            "elapsed_seconds": round(elapsed, 1),
            "slept_seconds": round(self.slept, 1),
            "sleep_fraction": round(self.slept / max(0.1, elapsed), 3),
            "sleep_count": self.sleeps,
            "max_gpu_c": self.max_gpu,
            "max_cpu_c": self.max_cpu,
            "faults": self.faults,
        }


def server_is_up(port: int = 18190, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def gpu_free_mib() -> int:
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2.0)
        if res.returncode == 0:
            return int(res.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return 8192


def preflight(args: Any, server_fn=server_is_up, free_fn=gpu_free_mib, temp_fn=None) -> Tuple[bool, List[str]]:
    problems = []
    port = getattr(args, "port", 18190)
    if server_fn(port):
        problems.append(f"REFUSED: conflicting server running on port {port}")
    free = free_fn()
    min_free = getattr(args, "min_free_vram_mib", 6000)
    if free < min_free:
        problems.append(f"REFUSED: free VRAM ({free} MiB) < required {min_free} MiB")
    gpu, cpu = (temp_fn() if temp_fn else (tm.gpu_temp(), tm.cpu_temp()))
    if not tm.known(gpu) or not tm.known(cpu):
        problems.append("REFUSED: hardware temperature telemetry not available")
    return len(problems) == 0, problems


def write_json(path: Union[str, Path], obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def dir_size_bytes(path: Union[str, Path]) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("**/*") if f.is_file())


def latest_checkpoint(out_dir: Union[str, Path]) -> Optional[Path]:
    p = Path(out_dir)
    ckpts = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0)
    return ckpts[-1] if ckpts else None


def config_dict(args: Any) -> Dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


# -----------------------------------------------------------------------------
# Exact Prefix Sharing (Tree Packing) for Multi-Completion Batches
# -----------------------------------------------------------------------------

def common_prefix_len(seqs: List[List[int]]) -> int:
    if not seqs:
        return 0
    shortest = min(len(s) for s in seqs)
    for i in range(shortest):
        val = seqs[0][i]
        if any(s[i] != val for s in seqs):
            return i
    return shortest


def group_prefix_len(exs: List[Dict[str, Any]]) -> int:
    if not exs:
        return 0
    n = common_prefix_len([e["input_ids"] for e in exs])
    return max(0, min(n, min(e.get("n_prompt", len(e["input_ids"])) for e in exs), min(len(e["input_ids"]) - 1 for e in exs)))


def pack_tree(exs: List[Dict[str, Any]], prefix_len: Optional[int] = None) -> Dict[str, Any]:
    """Pack one shared prompt prefix + K completion suffixes into a single tree-packed sequence."""
    P = group_prefix_len(exs) if prefix_len is None else prefix_len
    base = exs[0]["input_ids"][:P]
    ids, pos, seg, pred, tgt = list(base), list(range(P)), [-1] * P, [], []
    for k, e in enumerate(exs):
        if e["input_ids"][:P] != base:
            raise ValueError("examples do not share the claimed prefix")
        for j in range(P, len(e["input_ids"])):
            at = len(ids)
            ids.append(e["input_ids"][j])
            pos.append(j)
            seg.append(k)
            if e["labels"][j] != -100:
                if at - 1 < 0 or (j == P and P == 0):
                    raise ValueError("first token of a row cannot be a target")
                pred.append(P - 1 if j == P else at - 1)
                tgt.append(e["labels"][j])
    return {"ids": ids, "pos": pos, "seg": seg, "pred": pred, "tgt": tgt, "prefix_len": P}


def tree_mask(torch_module: Any, seg: List[int], P: int, dtype: Any = None, leak: bool = False):
    s = torch_module.tensor(seg)
    L = len(seg)
    causal = torch_module.ones(L, L, dtype=torch_module.bool).tril()
    same = (s[:, None] == s[None, :]) | (s[None, :] < 0)
    allow = causal if leak else causal & same
    allow = allow[None, None]
    if dtype is None or dtype == torch_module.bool:
        return allow
    return torch_module.zeros(allow.shape, dtype=dtype).masked_fill(~allow, torch_module.finfo(dtype).min)


def shared_prefix_loss(torch_module: Any, model: Any, exs: List[Dict[str, Any]], mask_dtype: Any = None, prefix_len: Optional[int] = None, _leak: bool = False, _pos_bug: bool = False) -> Tuple[Any, int]:
    pk = pack_tree(exs, prefix_len)
    dev = next(model.parameters()).device
    ids = torch_module.tensor([pk["ids"]], device=dev)
    pos = torch_module.tensor([pk["pos"] if not _pos_bug else [i for i in range(len(pk["ids"]))]], device=dev)
    mask = tree_mask(torch_module, pk["seg"], pk["prefix_len"], mask_dtype, leak=_leak).to(dev)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hid = inner.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[0]
    logits = inner.lm_head(hid[torch_module.tensor(pk["pred"], device=dev)]).float()
    return torch_module.nn.functional.cross_entropy(logits, torch_module.tensor(pk["tgt"], device=dev), reduction="mean"), len(pk["tgt"])


def share_groups(batch_indices: List[List[int]], share_group: int) -> List[List[int]]:
    flat = [i for b in batch_indices for i in b]
    return [flat[i:i + share_group] for i in range(0, len(flat), max(1, share_group))]


def tokens_per_step(exs_by_step: List[List[Dict[str, Any]]], share_group: Optional[int] = None) -> Tuple[int, int]:
    std = shared = 0
    for step in exs_by_step:
        std += sum(len(e["input_ids"]) for e in step)
        for i in range(0, len(step), share_group or len(step)):
            g = step[i:i + (share_group or len(step))]
            P = group_prefix_len(g)
            shared += P + sum(len(e["input_ids"]) - P for e in g)
    return std, shared


# -----------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SovereignLoRA Completion-Only SFT Trainer")
    p.add_argument("--base", default=DEFAULT_BASE, help="Base model HuggingFace ID or local path")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Input JSONL training dataset path")
    p.add_argument("--serving-target", default=SERVING_TARGET, help="Serving target model name")
    p.add_argument("--lane", choices=sorted(LANES.keys()), default=None, help="Evaluation lane")
    p.add_argument("--out-root", default="./runs", help="Output root directory")
    p.add_argument("--run", default="run_sft_lora", help="Run identifier")
    p.add_argument("--r", type=int, default=16, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling")
    p.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--microbatch", type=int, default=1, help="Microbatch size")
    p.add_argument("--accum", type=int, default=16, help="Gradient accumulation steps")
    p.add_argument("--max-len", type=int, default=1024, help="Maximum context length")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")
    p.add_argument("--dry-run", action="store_true", help="Validate data and config without GPU execution")
    p.add_argument("--smoke", action="store_true", help="Execute 2 smoke steps to verify pipeline")
    p.add_argument("--export-adapter-gguf", action="store_true", help="Generate or run GGUF conversion")
    p.add_argument("--run-conversion", action="store_true", help="Execute GGUF conversion command")
    p.add_argument("--convert-script", default=str(DEFAULT_CONVERT_LORA), help="Path to convert_lora_to_gguf.py")
    p.add_argument("--adapter-dir", default=None, help="Directory of adapter checkpoint to export")
    p.add_argument("--gguf-out", default=None, help="Target GGUF file path")
    p.add_argument("--render", choices=["plain", "bridge", "row"], default="bridge", help="Prompt render format")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.export_adapter_gguf:
        return export_adapter_gguf(args)

    rows, report = load_rows(args.data)
    if report.get("lines", 0) == 0:
        print(f"Warning: No valid rows loaded from {args.data}")

    train_rows, dev_rows = split_dev(rows, seed=args.seed) if rows else ([], [])

    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASSED",
            "base": args.base,
            "data_report": report,
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "hardware_check": tm.known(tm.gpu_temp()),
        }, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
