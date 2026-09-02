# ragged-batch

A from-scratch inference engine for a small open-weight decoder-only model, built in phases to demonstrate — and benchmark — the core mechanisms production LLM serving stacks (vLLM, TGI, TensorRT-LLM) rely on: KV caching, continuous batching, and quantization.

Each phase is independently runnable and its own commit, so the repo's history narrates the optimization story: naive → cached → batched → continuous → quantized, with a measured delta at each step.

## Model

[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — small enough to iterate fast and run on modest hardware, while still giving meaningful numbers.

## Phases

- [x] **Phase 0 — naive baseline.** Plain autoregressive decoding via a raw forward pass: recompute the full sequence every step, no cache. The correctness oracle every later phase must match token-for-token. → [`scripts/phase0_naive.py`](scripts/phase0_naive.py)
- [x] **Phase 1 — KV cache.** Cache per-layer K/V tensors, feed only the newest token each step. → [`scripts/phase1_kv_cache.py`](scripts/phase1_kv_cache.py)
- [ ] **Phase 2 — static batching.** Batch several prompts into one padded forward pass; surfaces static batching's head-of-line blocking flaw before fixing it.
- [ ] **Phase 3 — continuous batching.** A request queue plus a decode-step-granularity scheduler with per-sequence (ragged) KV caches — the centerpiece this repo is named for.
- [ ] **Phase 4 — quantization.** Weight-only INT8 (and optionally INT4/NF4) on the linear layers.
- [ ] **Phase 5 — benchmark + write-up.** Consolidated latency/throughput plots across all variants.

## Setup

Requires Python ≥3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running

```bash
uv run python scripts/phase0_naive.py
```

Downloads and caches the model weights (~1GB) from Hugging Face Hub on first run.

## License

[MIT](LICENSE)
