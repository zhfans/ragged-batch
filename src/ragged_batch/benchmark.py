"""Decode-timing instrumentation and the token-equivalence check every phase is measured against."""

import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from ragged_batch.device import synchronize


@dataclass
class DecodeTrace:
    """Wall-clock timings from one greedy-decode run.

    ``prefill_s`` is the first forward pass (the whole prompt); ``decode_step_s``
    holds one entry per subsequent single-token step. Keeping the steps
    un-aggregated lets a caller compare per-token latency across prompt lengths,
    which is the Phase 0 vs Phase 1 story.
    """

    device: str
    prompt_tokens: int
    prefill_s: float
    decode_step_s: list[float]

    @property
    def decode_steps(self) -> int:
        """Timed single-token steps. One fewer than tokens generated — the prefill pass emits the first."""
        return len(self.decode_step_s)

    @property
    def decode_s(self) -> float:
        return sum(self.decode_step_s)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def decode_tokens_per_s(self) -> float:
        return self.decode_steps / self.decode_s if self.decode_s else float("nan")

    @property
    def median_step_ms(self) -> float:
        return _percentile(self.decode_step_s, 50) * 1e3

    @property
    def p99_step_ms(self) -> float:
        return _percentile(self.decode_step_s, 99) * 1e3

    def summary(self) -> str:
        """One-line human-readable digest for logging."""
        return (
            f"prefill {self.prefill_s * 1e3:.1f} ms · "
            f"{self.decode_steps} steps @ {self.decode_tokens_per_s:.1f} tok/s · "
            f"step median {self.median_step_ms:.1f} ms p99 {self.p99_step_ms:.1f} ms · "
            f"prompt {self.prompt_tokens} tok ({self.device})"
        )


class DecodeTimer:
    """Collects prefill and per-step timings, synchronizing the device around each.

    Time the prompt pass inside :meth:`prefill` and each decode step inside
    :meth:`step`, then call :meth:`build`::

        timer = DecodeTimer(device)
        with timer.prefill():
            out = model(input_ids, use_cache=True)
        for _ in range(max_new_tokens):
            with timer.step():
                out = model(next_token, past_key_values=cache, use_cache=True)
            ...
        trace = timer.build(prompt_tokens=input_ids.shape[1])
    """

    def __init__(self, device: str) -> None:
        self._device = device
        self._prefill_s: float | None = None
        self._decode_step_s: list[float] = []

    @contextmanager
    def prefill(self) -> Iterator[None]:
        synchronize(self._device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self._device)
            self._prefill_s = time.perf_counter() - start

    @contextmanager
    def step(self) -> Iterator[None]:
        synchronize(self._device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self._device)
            self._decode_step_s.append(time.perf_counter() - start)

    def build(self, *, prompt_tokens: int) -> DecodeTrace:
        if self._prefill_s is None:
            raise RuntimeError("prefill() context was never entered")
        return DecodeTrace(
            device=self._device,
            prompt_tokens=prompt_tokens,
            prefill_s=self._prefill_s,
            decode_step_s=list(self._decode_step_s),
        )


def assert_tokens_match(
    reference: Sequence[int],
    candidate: Sequence[int],
    *,
    reference_name: str = "phase 0",
    candidate_name: str = "candidate",
) -> None:
    """Raise ``AssertionError`` unless two token-id sequences are identical.

    The Phase 1+ invariant: an optimization may only change speed, never the
    tokens greedy decode emits. The message points at the first divergence, where
    a RoPE/position bug usually shows up.
    """
    ref = list(reference)
    cand = list(candidate)
    if ref == cand:
        return
    if len(ref) != len(cand):
        detail = f"lengths differ: {len(ref)} vs {len(cand)}"
    else:
        i = next(k for k in range(len(ref)) if ref[k] != cand[k])
        detail = f"first divergence at index {i}: {ref[i]} vs {cand[i]}"
    raise AssertionError(
        f"{candidate_name} tokens do not match {reference_name} ({detail})"
    )


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile; ``nan`` for an empty input."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)
