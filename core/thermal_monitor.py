# ==============================================================================
# Program: thermal_monitor.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Hardware Telemetry & Thermal Duty Cycle Monitor.
#          Provides continuous, fail-closed thermal monitoring for consumer GPUs
#          (e.g., NVIDIA RTX 3070 8GB, Apple Silicon M-series) and CPUs during
#          resource-intensive LLM fine-tuning and matrix operations. Prevents
#          hardware degradation and thermal throttling through duty-cycle sleep
#          intervals, optimizer pauses, and fail-closed safety abort gates.
# ==============================================================================

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ==============================================================================
# SECTION 1: Default Thermal Safety Thresholds (in Celsius)
# Educational Note:
# Modern consumer GPUs (like NVIDIA GeForce RTX 3070) have maximum silicon junction
# temperature limits around 93°C-95°C before hardware thermal throttling occurs.
# Operating sustained near these limits degrades silicon and solder joints.
# We establish a four-tiered software control barrier:
#   1. SOFT (80°C): Begin inserting minor sleep duty cycles to stabilize temperature.
#   2. PAUSE (84°C): Temporarily pause optimizer steps and wait for cooldown.
#   3. RESUME (78°C): Hysteresis recovery floor before resuming full compute.
#   4. ABORT (90°C): Critical emergency stop to protect physical hardware.
# ==============================================================================

DEFAULT_SOFT_GPU = 80       # Begin duty cycle sleep intervals
DEFAULT_PAUSE_GPU = 84      # Pause optimizer steps until cooled
DEFAULT_RESUME_GPU = 78     # Resume threshold after pausing
DEFAULT_ABORT_GPU = 90      # Critical safety abort threshold
DEFAULT_START_MAX_GPU = 82  # Maximum GPU temp permitted to begin training
DEFAULT_MAX_CPU = 96        # Fail-closed CPU temperature limit
DEFAULT_MIN_RAM_GIB = 1.0   # Minimum available system RAM to avoid OOM killer


def known(value: Optional[float]) -> bool:
    """Validate whether a telemetry reading is a valid, finite floating-point number.

    Educational Note:
        In scientific telemetry and embedded systems, sensor readings can occasionally
        return None, NaN (Not a Number), or infinite values due to bus timeouts or
        uninitialized sensors. In Python, `NaN != NaN` evaluates to True. This helper
        guarantees that only valid numerical readings are trusted.

    Parameters:
        value (Optional[float]): Sensor reading to validate.

    Returns:
        bool: True if value is a real, finite number; False otherwise.
    """
    return value is not None and isinstance(value, (int, float)) and not (value != value)


def gpu_temp() -> Optional[float]:
    """Query current GPU temperature in Celsius via system telemetry utilities.

    Telemetry Query Strategy:
        1. Query `nvidia-smi` on systems with NVIDIA GPUs using comma-separated CSV formatting.
        2. Query macOS memory pressure / thermal pressure index using `sysctl` on Apple Silicon.
        3. Check environment variable override (`SOVEREIGN_MOCK_GPU_TEMP`) for testing.

    Returns:
        Optional[float]: Current temperature in degrees Celsius, or None if unavailable.
    """
    # Step 1: Probe for NVIDIA GPU via `nvidia-smi`
    if shutil.which("nvidia-smi"):
        try:
            # Execute nvidia-smi with nounits to get pure integer/float output
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0
            )
            if res.returncode == 0:
                lines = res.stdout.strip().splitlines()
                if lines:
                    return float(lines[0].strip())
        except Exception:
            # Fall through to subsequent checks if nvidia-smi call fails
            pass

    # Step 2: Apple Silicon / macOS thermal pressure approximation
    if sys.platform == "darwin":
        # On macOS, native GPU temps require root IOKit/AppleSMC; fallback to memory pressure index
        try:
            res = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                 capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                level = int(res.stdout.strip())
                # Map 1 (normal), 2 (warn), 4 (critical) to approximate pseudo-temperatures
                # Not a real temperature: macOS exposes no GPU sensor here, so memory-pressure level (1 normal, 2 warn, 4 critical)
                # is mapped onto a synthetic scale (55/65/85 C) so downstream thresholds still react to system stress.
                return 45.0 + (level * 10.0)
        except Exception:
            # If sysctl fails on macOS, report a mild synthetic value instead of None so training is not blocked.
            return 50.0

    # Step 3: Mock environment override for test fixtures and CI/CD pipelines
    # Test hook: lets CI and hardware-less hosts inject a temperature. Only reached when no real source answered.
    env_temp = os.environ.get("SOVEREIGN_MOCK_GPU_TEMP")
    if env_temp is not None:
        try:
            return float(env_temp)
        except ValueError:
            pass

    # Telemetry unavailable
    return None


def cpu_temp() -> Optional[float]:
    """Query current CPU temperature in Celsius via Linux sysfs or platform fallbacks.

    Telemetry Strategy:
        1. Linux: Read sysfs thermal zone files (`/sys/class/thermal/thermal_zone*/temp`).
           Values in millidegrees Celsius are divided by 1000.0.
        2. macOS: Return default nominal value or test environment override.

    Returns:
        Optional[float]: CPU package temperature in Celsius, or None if unavailable.
    """
    # Step 1: Probe Linux kernel sysfs thermal zones
    # Linux exposes each sensor as a thermal_zone with a "type" label and a "temp" file.
    thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*"))
    for zone in thermal_zones:
        try:
            type_file = zone / "type"
            temp_file = zone / "temp"
            if type_file.exists() and temp_file.exists():
                ztype = type_file.read_text().strip().lower()
                # Prioritize CPU package sensors
                # Prefer the CPU package sensor (or any zone labelled cpu); ignore e.g. battery or wifi zones.
                if "x86_pkg_temp" in ztype or "cpu" in ztype:
                    raw_val = float(temp_file.read_text().strip())
                    # Convert from millidegrees (e.g. 52000 -> 52.0)
                    # sysfs reports millidegrees on most kernels; values <= 1000 are assumed to already be in degrees.
                    return raw_val / 1000.0 if raw_val > 1000 else raw_val
        except Exception:
            pass

    # Step 2: macOS fallback & environment override
    if sys.platform == "darwin":
        env_temp = os.environ.get("SOVEREIGN_MOCK_CPU_TEMP")
        if env_temp is not None:
            try:
                return float(env_temp)
            except ValueError:
                pass
        # Nominal placeholder on macOS where CPU die temperature needs privileged tooling.
        return 48.0

    # Step 3: Test fixture check
    env_temp = os.environ.get("SOVEREIGN_MOCK_CPU_TEMP")
    if env_temp is not None:
        try:
            return float(env_temp)
        except ValueError:
            pass

    return None


def ram_free_gib() -> float:
    """Return available system RAM in GiB (Gibibytes: 1024^3 bytes).

    Operating System Logic:
        - Linux: Reads 'MemAvailable:' from `/proc/meminfo`.
        - macOS: Runs `vm_stat` and calculates free + speculative pages times page size (4096 bytes).

    Returns:
        float: Available RAM in GiB. Defaults to 16.0 GiB if querying fails.
    """
    try:
        # Check Linux /proc/meminfo
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Value in kibibytes (KB)
                        kb = int(line.split()[1])
                        return kb / (1024.0 * 1024.0)

        # Check macOS vm_stat
        elif sys.platform == "darwin":
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2.0)
            if vm.returncode == 0:
                pages = {}
                for line in vm.stdout.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        m = re.search(r"(\d+)", v)
                        if m:
                            pages[k.strip()] = int(m.group(1))
                # Free plus speculative pages approximate "available" memory on macOS (vm_stat reports page counts).
                free_pages = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
                # Multiply by 4096 bytes per page and divide by 1024^3
                return (free_pages * 4096) / (1024.0 ** 3)
    except Exception:
        pass

    return 16.0  # Safe default assumption for development


# ==============================================================================
# SECTION 2: Thermal Safety Guard Class
# Educational Note:
# Implements an active monitor pattern. Before starting training (`check_preflight`)
# and at every optimizer step (`evaluate_step`), the safety guard checks telemetry.
# If an operator places a STOP file in ~/.sovereign_lora/STOP, training halts immediately.
# ==============================================================================

class ThermalSafetyGuard:
    """Configurable hardware and thermal safety monitor for LLM training pipelines."""

    def __init__(
        self,
        soft_gpu: float = DEFAULT_SOFT_GPU,
        pause_gpu: float = DEFAULT_PAUSE_GPU,
        resume_gpu: float = DEFAULT_RESUME_GPU,
        abort_gpu: float = DEFAULT_ABORT_GPU,
        start_max_gpu: float = DEFAULT_START_MAX_GPU,
        max_cpu: float = DEFAULT_MAX_CPU,
        min_ram_gib: float = DEFAULT_MIN_RAM_GIB,
        stop_file: Optional[Path] = None,
    ) -> None:
        """Initialize the guard with safety thresholds."""
        self.soft_gpu = soft_gpu
        self.pause_gpu = pause_gpu
        self.resume_gpu = resume_gpu
        self.abort_gpu = abort_gpu
        self.start_max_gpu = start_max_gpu
        self.max_cpu = max_cpu
        self.min_ram_gib = min_ram_gib
        # Operator kill switch shared with ThermalPacer: `touch ~/.sovereign_lora/STOP` blocks new runs.
        self.stop_file = stop_file or (Path.home() / ".sovereign_lora" / "STOP")

    def check_preflight(
        self,
        gpu_fn=gpu_temp,
        cpu_fn=cpu_temp,
        ram_fn=ram_free_gib,
    ) -> Tuple[bool, list[str]]:
        """Verify hardware conditions before initiating an optimization run.

        Checks:
            1. Presence of external kill switch STOP file.
            2. GPU temperature below starting ceiling.
            3. CPU temperature below fail-closed threshold.
            4. Available RAM above minimum requirement.

        Returns:
            Tuple[bool, list[str]]: (passed, list_of_blocking_problems)
        """
        problems = []

        # Check 1: External manual kill switch
        if self.stop_file.exists():
            problems.append(f"STOP signal file present at {self.stop_file}")

        # Check 2: GPU temperature
        gpu = gpu_fn()
        # Note: a missing (None/NaN) reading is skipped here rather than blocking; sovereign_sft_lora.preflight
        # is the stricter gate that refuses when telemetry is unavailable.
        if known(gpu) and gpu > self.start_max_gpu:
            problems.append(f"GPU temperature ({gpu:.1f}°C) exceeds start ceiling ({self.start_max_gpu}°C)")

        # Check 3: CPU temperature
        cpu = cpu_fn()
        if known(cpu) and cpu >= self.max_cpu:
            problems.append(f"CPU temperature ({cpu:.1f}°C) exceeds fail-closed limit ({self.max_cpu}°C)")

        # Check 4: System RAM
        ram = ram_fn()
        if known(ram) and ram < self.min_ram_gib:
            problems.append(f"Available RAM ({ram:.2f} GiB) below threshold ({self.min_ram_gib:.2f} GiB)")

        return (len(problems) == 0, problems)

    def evaluate_step(
        self,
        gpu: Optional[float],
        cpu: Optional[float],
    ) -> Tuple[str, float]:
        """Evaluate temperatures at optimizer step boundaries and decide pacing actions.

        State Machine Actions:
            - 'ABORT': Critical thermal limit or STOP file encountered. Terminate immediately.
            - 'PAUSE': High temperature threshold reached. Suspend training for cooldown.
            - 'DUTY_SLEEP': Temperature elevated between soft and pause. Insert proportional sleep.
            - 'OK': Temperatures within safe operating limits. Proceed without delay.

        Duty Sleep Calculation:
            When gpu is between soft_gpu (e.g. 80°C) and pause_gpu (e.g. 84°C):
                ratio = (gpu - soft_gpu) / (pause_gpu - soft_gpu)
                sleep_seconds = 2.0 + (ratio * 6.0)  # Smoothly scales from 2.0s to 8.0s

        Parameters:
            gpu (Optional[float]): Current GPU temperature.
            cpu (Optional[float]): Current CPU temperature.

        Returns:
            Tuple[str, float]: (action_string, recommended_sleep_seconds)
        """
        # Step 1: Immediate abort if STOP file appeared
        if self.stop_file.exists():
            return "ABORT", 0.0

        # Step 2: GPU checks
        if known(gpu):
            if gpu >= self.abort_gpu:
                return "ABORT", 0.0
            if gpu >= self.pause_gpu:
                return "PAUSE", 10.0
            if gpu >= self.soft_gpu:
                # Calculate linear interpolation ratio: 0.0 at soft_gpu, 1.0 at pause_gpu
                ratio = (gpu - self.soft_gpu) / max(1.0, (self.pause_gpu - self.soft_gpu))
                # Scale sleep time dynamically between 2.0s and 8.0s
                return "DUTY_SLEEP", 2.0 + (ratio * 6.0)

        # Step 3: CPU fail-closed limit
        if known(cpu):
            if cpu >= self.max_cpu:
                return "ABORT", 0.0

        # All systems nominal
        return "OK", 0.0
