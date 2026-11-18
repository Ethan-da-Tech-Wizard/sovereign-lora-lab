# ==============================================================================
# Program: __init__.py (core package)
# Programmer: Ethan-da-Tech-Wizard
# Course: CIS 156 / Advanced Systems
# Last Modified: November 18, 2026
# Purpose: Top-level package exports for SovereignLoRA core engine modules:
#          - Thermal telemetry and duty cycle pacing (thermal_monitor)
#          - Verifiable reinforcement learning rewards & SQLite logging (grpo_reward_scorer)
#          - Multi-LoRA SVD and TIES fusion (adapter_merger)
#          - Autonomous DSCE Oracle & DPO preference harvesting (autonomous_dsce_trainer)
# ==============================================================================

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
