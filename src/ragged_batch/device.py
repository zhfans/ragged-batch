"""Device selection and synchronization for accurate wall-clock timing."""

import torch


def pick_device() -> str:
    """Return the fastest available torch device: ``cuda`` > ``mps`` > ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    """Block until all queued work on ``device`` has finished.

    Kernel launches on CUDA and MPS are asynchronous, so a ``time.perf_counter()``
    reading taken right after a forward pass would otherwise capture only the
    launch, not the compute. A no-op on CPU.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
