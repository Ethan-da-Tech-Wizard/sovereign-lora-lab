# Program: sovereign-lora-lab - __init__.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Package entry point that re-exports the thermal, GRPO reward, adapter merging and DSCE trainer helpers so callers can import them from core.

"""SovereignLoRA Core Package."""

from core.thermal_monitor import (
    ThermalSafetyGuard,
    gpu_temp,
    cpu_temp,
    known,
)
from core.grpo_reward_scorer import (
    evaluate_command_reward,
    evaluate_code_reward,
    score_dpo_pair,
    score_and_log_event,
)
from core.adapter_merger import (
    LoRAAdapter,
    linear_task_arithmetic,
    truncated_svd_merge,
    ties_merge,
    dare_sparsify,
)
from core.autonomous_dsce_trainer import (
    AutonomousDSCETrainer,
    query_dsce_oracle,
)

__all__ = [
    "ThermalSafetyGuard",
    "gpu_temp",
    "cpu_temp",
    "known",
    "evaluate_command_reward",
    "evaluate_code_reward",
    "score_dpo_pair",
    "score_and_log_event",
    "LoRAAdapter",
    "linear_task_arithmetic",
    "truncated_svd_merge",
    "ties_merge",
    "dare_sparsify",
    "AutonomousDSCETrainer",
    "query_dsce_oracle",
]
