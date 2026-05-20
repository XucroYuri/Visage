"""Hardware detection and adaptive resource configuration.

Probes macOS system resources (RAM, CPU cores) and translates them into
processing parameter recommendations so the pipeline stays within hardware
limits even on constrained devices.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    total_ram_gb: float
    available_ram_gb: float
    physical_cores: int
    logical_cores: int


@dataclass
class ResourceConfig:
    """Recommended processing parameters based on hardware."""

    backend: str = "insightface"  # "dlib" | "insightface"
    max_workers: int = 4
    max_image_dimension: int = 0  # 0 = no downscaling
    use_float32_cluster: bool = False
    cluster_chunk_size: int = 0  # 0 = no chunking (full NxN matrix)
    head_feature_weight: float = 0.2
    sample_limit: int | None = None  # max faces before sampling


def _sysctl(key: str) -> str:
    """Run sysctl and return the value. Raises RuntimeError on failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"sysctl {key} failed: {e}") from e
    return result.stdout.strip()


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities via sysctl on macOS.

    Returns:
        HardwareProfile with total RAM, estimated available RAM, and core counts.
    """
    # Physical RAM
    try:
        mem_bytes = int(_sysctl("hw.memsize"))
    except RuntimeError:
        logger.warning("Cannot detect RAM via sysctl, assuming 4 GB")
        mem_bytes = 4 * 1024**3

    total_ram_gb = mem_bytes / (1024**3)

    # Available RAM estimate: subtract a fixed OS overhead (~1.5 GB)
    # and account for other running processes
    os_overhead_gb = 1.5
    available_ram_gb = max(0.5, total_ram_gb - os_overhead_gb)

    # CPU cores
    try:
        physical_cores = int(_sysctl("hw.physicalcpu"))
    except RuntimeError:
        physical_cores = 1

    try:
        logical_cores = int(_sysctl("hw.logicalcpu"))
    except RuntimeError:
        logical_cores = max(1, physical_cores)

    return HardwareProfile(
        total_ram_gb=round(total_ram_gb, 1),
        available_ram_gb=round(available_ram_gb, 1),
        physical_cores=physical_cores,
        logical_cores=logical_cores,
    )


def recommend_config(
    profile: HardwareProfile,
    face_count_hint: int | None = None,
) -> ResourceConfig:
    """Recommend processing parameters based on hardware profile.

    Tier logic:
        >= 16 GB RAM → full InsightFace, no constraints
        >= 8 GB  RAM → InsightFace with conservative tuning
        >= 4 GB  RAM → force dlib, float32, chunked clustering
        < 4 GB   RAM → dlib + DBSCAN (no HDBSCAN)

    Args:
        profile: Hardware profile from detect_hardware().
        face_count_hint: Estimated face count (if known) for tuning cluster params.

    Returns:
        ResourceConfig with recommended parameters.
    """
    ram = profile.total_ram_gb
    cores = profile.physical_cores

    if ram >= 16:
        cfg = ResourceConfig(
            backend="insightface",
            max_workers=min(cores, 8),
            use_float32_cluster=False,
            head_feature_weight=0.2,
        )
    elif ram >= 8:
        cfg = ResourceConfig(
            backend="insightface",
            max_workers=min(cores, 4),
            max_image_dimension=2048,
            use_float32_cluster=True,
            head_feature_weight=0.15,
        )
    elif ram >= 4:
        cfg = ResourceConfig(
            backend="dlib",
            max_workers=min(cores, 2),
            max_image_dimension=1536,
            use_float32_cluster=True,
            head_feature_weight=0.0,
        )
    else:
        cfg = ResourceConfig(
            backend="dlib",
            max_workers=1,
            max_image_dimension=1024,
            use_float32_cluster=True,
            head_feature_weight=0.0,
        )

    # Enable chunked clustering for large face counts on constrained RAM
    if face_count_hint is not None and face_count_hint > 2000:
        if ram < 8 or cfg.use_float32_cluster:
            cfg.cluster_chunk_size = 500 if ram >= 4 else 250

    # Use sampling for extreme datasets
    if face_count_hint is not None and face_count_hint > 10000:
        cfg.sample_limit = 8000 if ram >= 8 else 4000
        if ram < 8:
            cfg.cluster_chunk_size = 250

    return cfg


def memory_pressure_ok(threshold_gb: float = 0.5) -> bool:
    """Check if available system memory is above the threshold.

    Uses vm_stat on macOS to estimate current memory pressure.
    Returns True if there's enough free memory, False if pressure is high.

    Args:
        threshold_gb: Minimum acceptable free memory in GB.
    """
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Cannot check — assume ok
        return True

    vm = {}
    for line in result.stdout.splitlines():
        parts = line.strip().rstrip(".").split(":")
        if len(parts) < 2:
            continue
        key = parts[0].strip().strip('"')
        try:
            vm[key] = int(parts[1].strip())
        except ValueError:
            continue

    page_size = vm.get("page size of 4K", 4096)
    free_pages = vm.get("Pages free", 0)
    # speculative pages can be reclaimed
    speculative_pages = vm.get("Pages speculative", 0)

    free_gb = ((free_pages + speculative_pages) * page_size) / (1024**3)
    return free_gb >= threshold_gb
