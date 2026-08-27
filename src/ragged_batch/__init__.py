"""Shared helpers for the ragged-batch phase scripts.

Each ``scripts/phaseN_*.py`` owns its decode loop; this package holds only what
they have in common: device selection, model loading, and the timing +
equivalence instrumentation their benchmarks report.
"""

from ragged_batch.benchmark import DecodeTimer, DecodeTrace, assert_tokens_match
from ragged_batch.device import pick_device, synchronize
from ragged_batch.model import DEFAULT_MODEL_ID, load_model

__all__ = [
    "DEFAULT_MODEL_ID",
    "DecodeTimer",
    "DecodeTrace",
    "assert_tokens_match",
    "load_model",
    "pick_device",
    "synchronize",
]
