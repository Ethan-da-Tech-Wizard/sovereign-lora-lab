# ==============================================================================
# Program: __init__.py (core package)
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Top-level package exports for SovereignLoRA core engine modules:
#          - Thermal telemetry and duty cycle pacing (thermal_monitor)
#          - Verifiable reinforcement learning rewards & SQLite logging (grpo_reward_scorer)
#          - Multi-LoRA SVD and TIES fusion (adapter_merger)
#          - Autonomous DSCE Oracle & DPO preference harvesting (autonomous_dsce_trainer)
# ==============================================================================

"""SovereignLoRA Core Package."""

# Re-export the public API so callers can write `from core import ThermalSafetyGuard, ...`.
# Thermal telemetry and safety (core/thermal_monitor.py): sensor readers plus the preflight / per-step guard
# used before and during training.
from core.thermal_monitor import (
    ThermalSafetyGuard,
    gpu_temp,
    cpu_temp,
    known,
)
# Deterministic reward functions for GRPO/DPO (core/grpo_reward_scorer.py): score a shell command or Python snippet
# and log the event to SQLite. No model is involved, so rewards are reproducible.
from core.grpo_reward_scorer import (
    evaluate_command_reward,
    evaluate_code_reward,
    score_dpo_pair,
    score_and_log_event,
)
# Pure-Python LoRA adapter fusion (core/adapter_merger.py): linear task arithmetic, SVD re-compression, TIES and DARE.
from core.adapter_merger import (
    LoRAAdapter,
    linear_task_arithmetic,
    truncated_svd_merge,
    ties_merge,
    dare_sparsify,
)
# Oracle-verified preference harvesting (core/autonomous_dsce_trainer.py). Note: sovereign_sft_lora is
# not re-exported here; import it directly (core.sovereign_sft_lora) so its helpers load only when needed.
from core.autonomous_dsce_trainer import (
    AutonomousDSCETrainer,
    query_dsce_oracle,
)

# Names exported by `from core import *`; keep in sync with the imports above.
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
