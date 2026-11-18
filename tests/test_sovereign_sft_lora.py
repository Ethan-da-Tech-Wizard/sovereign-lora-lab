#!/usr/bin/env python3
"""tests/test_sovereign_sft_lora.py — Standalone unit tests for SovereignLoRA SFT training engine."""

import json
import tempfile
import unittest
from pathlib import Path

from core import sovereign_sft_lora as m


def sample_rows(n=40):
    out = []
    for i in range(n):
        out.append({
            "prompt": f"prompt_{i}",
            "chosen": json.dumps({"action": i}) if i % 5 else "REFUSE",
            "family": f"family_{i % 4}",
        })
    out.append(dict(out[3]))  # Exact duplicate
    return out


class FakeTokenizer:
    eos_token = "<|im_end|>"
    chat_template = "x"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True, **kw):
        s = "".join(f"<|im_start|>{x['role']}\n{x.get('content') or json.dumps(x.get('tool_calls'))}<|im_end|>\n" for x in msgs)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}


class TestSovereignSFTLoRA(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.d = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_dataset(self, rs):
        p = self.d / "test_pairs.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rs))
        return p

    def test_load_drop_dedupe_and_hash_guard(self):
        rs = sample_rows(40)
        p = self.write_dataset(rs)
        kept, rep = m.load_rows(p, drop_ids=(0, 1))
        self.assertEqual(rep["exact_duplicates_removed"], 1)
        self.assertEqual(len(kept), 38)

        # Hash guard mismatch fails closed
        with self.assertRaises(ValueError):
            m.load_rows(p, drop_ids=(0,), drop_hashes={0: "badhash1234"})

        # Lenient hash mismatch keeps row with warning
        kept_lenient, rep_lenient = m.load_rows(p, drop_ids=(0,), drop_hashes={0: "badhash1234"}, lenient_hash=True)
        self.assertEqual(rep_lenient["drop_hash_mismatch_not_dropped"], 1)

    def test_split_deterministic_disjoint(self):
        rs = sample_rows(40)
        p = self.write_dataset(rs)
        rows, _ = m.load_rows(p)
        train_a, dev_a = m.split_dev(rows, seed=1101, dev_n=8)
        train_b, dev_b = m.split_dev(rows, seed=1101, dev_n=8)

        self.assertEqual([x["idx"] for x in dev_a], [x["idx"] for x in dev_b])
        self.assertEqual(len(dev_a), 8)

        # Disjoint prompts invariant
        train_prompts = {x["prompt"] for x in train_a}
        dev_prompts = {x["prompt"] for x in dev_a}
        self.assertEqual(len(train_prompts & dev_prompts), 0)

    def test_completion_only_mask(self):
        tok = FakeTokenizer()
        ex = m.encode_example(tok, "Hello world", "Hi there!", max_len=512)
        self.assertIsNotNone(ex)
        ids = ex["input_ids"]
        labels = ex["labels"]
        self.assertEqual(len(ids), len(labels))

        # First tokens should be masked with -100
        self.assertEqual(labels[0], -100)
        # Last token should be unmasked (completion token)
        self.assertNotEqual(labels[-1], -100)

    def test_schedule_and_lr(self):
        # 100 rows, microbatch 1, accum 4 -> 4 rows per step -> 25 steps per epoch -> 75 steps for 3 epochs
        sched = m.build_schedule(n=100, micro=1, accum=4, epochs=3, seed=42)
        self.assertEqual(len(sched), 75)

        lr_warmup = m.lr_at(step=2, total=100, warmup=10, peak=2e-4)
        self.assertAlmostEqual(lr_warmup, 2e-4 * (3 / 10))

        lr_peak = m.lr_at(step=10, total=100, warmup=10, peak=2e-4)
        self.assertAlmostEqual(lr_peak, 2e-4)

    def test_cooling_plan_and_fault(self):
        # Normal temp: no cooling
        target, fixed, cap, ckpt = m.cooling_plan(75.0)
        self.assertIsNone(target)
        self.assertEqual(fixed, 0)

        # Soft band: short sleep
        target, fixed, cap, ckpt = m.cooling_plan(81.0)
        self.assertEqual(fixed, 5)

        # Hot band: emergency cooling
        target, fixed, cap, ckpt = m.cooling_plan(88.0)
        self.assertEqual(target, 76)
        self.assertTrue(ckpt)

    def test_thermal_pacer_simulation(self):
        log_file = self.d / "pacer.jsonl"
        temps = [75.0, 78.0, 81.0, 76.0]
        temp_iter = iter(temps)

        def mock_gpu():
            return next(temp_iter, 75.0)

        pacer = m.ThermalPacer(
            log_path=log_file,
            gpu_fn=mock_gpu,
            cpu_fn=lambda: 60.0,
            sleep_fn=lambda s: None,
        )
        status = pacer.after_step(step=0)
        self.assertEqual(status, "ok")
        summ = pacer.summary()
        self.assertEqual(summ["max_gpu_c"], 75.0)

    def test_serving_check(self):
        # Target match
        ok, dec = m.serving_check("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct")
        self.assertTrue(ok)

        # Candidate lane
        ok_lane, dec_lane = m.serving_check("Qwen/Qwen2.5-Coder-1.5B", lane="candidate-1.5b")
        self.assertTrue(ok_lane)
        self.assertFalse(dec_lane["production_ready"])

    def test_estimate_resources(self):
        cfg = {"base_weights_gb": 3.8, "lora_weights_gb": 0.15}
        est = m.estimate_resources(
            cfg=cfg,
            max_len=1024,
            micro=1,
            n_rows=500,
            epochs=3,
            mean_len=256,
            mean_completion=64,
            chunked_logits=True,
        )
        self.assertTrue(est["fits_8gb_vram"])
        self.assertLess(est["vram_peak_gb"], 7.0)

    def test_pack_tree_and_prefix_sharing(self):
        shared_prefix = [10, 20, 30]
        ex1 = {"input_ids": shared_prefix + [101, 102], "labels": [-100, -100, -100, 101, 102], "n_prompt": 3}
        ex2 = {"input_ids": shared_prefix + [201, 202], "labels": [-100, -100, -100, 201, 202], "n_prompt": 3}

        P = m.group_prefix_len([ex1, ex2])
        self.assertEqual(P, 3)

        packed = m.pack_tree([ex1, ex2], prefix_len=P)
        self.assertEqual(packed["prefix_len"], 3)
        self.assertEqual(packed["ids"][:3], shared_prefix)
        # Suffix tokens appended
        self.assertEqual(packed["ids"][3:], [101, 102, 201, 202])

    def test_adapter_gguf_command(self):
        cmd = m.adapter_gguf_command("adapter_path", "base_path", "out.gguf")
        self.assertIn("convert_lora_to_gguf.py", cmd[1])
        self.assertIn("--base", cmd)
        self.assertIn("base_path", cmd)


if __name__ == "__main__":
    unittest.main()
