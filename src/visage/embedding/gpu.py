"""GPU device detection and auto-fallback for embedding inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Detected compute device information."""

    device: str  # "mps", "cuda", "cpu"
    name: str  # Human-readable device name
    supports_half: bool  # Whether float16 is reliable on this device

    @property
    def is_gpu(self) -> bool:
        return self.device in ("mps", "cuda")


def detect_device(prefer: str | None = None) -> DeviceInfo:
    """Detect the best available compute device.

    Priority: prefer (if specified) > cuda > mps > cpu.

    Args:
        prefer: Override device preference ("cuda", "mps", "cpu").

    Returns:
        DeviceInfo with the selected device.
    """
    if prefer == "cpu":
        return _cpu_device()
    if prefer == "cuda":
        info = _detect_cuda()
        if info:
            return info
        logger.warning("CUDA requested but not available, falling back to CPU")
        return _cpu_device()
    if prefer == "mps":
        info = _detect_mps()
        if info:
            return info
        logger.warning("MPS requested but not available, falling back to CPU")
        return _cpu_device()

    # Auto-detect: cuda > mps > cpu
    info = _detect_cuda()
    if info:
        return info
    info = _detect_mps()
    if info:
        return info
    return _cpu_device()


def _detect_cuda() -> DeviceInfo | None:
    """Check for CUDA-capable GPU via torch."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # Half precision is reliable on all CUDA devices >= sm_53
            capability = torch.cuda.get_device_capability(0)
            supports_half = capability >= (5, 3)
            return DeviceInfo(device="cuda", name=name, supports_half=supports_half)
    except ImportError:
        pass
    return None


def _detect_mps() -> DeviceInfo | None:
    """Check for Apple Metal Performance Shaders via torch."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(
                device="mps",
                name="Apple Metal Performance Shaders",
                supports_half=True,
            )
    except ImportError:
        pass
    return None


def _cpu_device() -> DeviceInfo:
    """Return CPU device info."""
    import platform

    return DeviceInfo(
        device="cpu",
        name=f"{platform.processor() or platform.machine()} (CPU)",
        supports_half=False,
    )
