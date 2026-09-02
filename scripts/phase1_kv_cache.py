"""Phase 1 — KV cache.

The same greedy decode as Phase 0, but each layer's K/V is kept across steps and
only the newest token is fed after the prompt. Tokens stay identical to Phase 0;
per-step work drops from O(n) to O(1) (plus the O(n) attention read over the
cache), so the latency gap widens with sequence length.
"""

import torch
from loguru import logger
from phase0_naive import naive_decode
from transformers import PreTrainedModel

from ragged_batch import (
    DecodeTimer,
    assert_tokens_match,
    load_model,
    pick_device,
    resolve_eos_id,
)

PROMPT = "Hello, my name is"
EQUIVALENCE_TOKENS = 64
GEN_LENGTHS = (32, 128, 512)


@torch.no_grad()
def cached_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None,
    timer: DecodeTimer,
) -> torch.Tensor:
    """Greedy-decode with a growing KV cache; return prompt + generated ids.

    The prefill pass consumes the whole prompt and fills the cache, yielding the
    first token. Each decode step then feeds only that last token (shape
    ``[1, 1]``) with ``past_key_values``, so the model runs its projections and
    MLP over one position instead of the whole sequence.
    """
    with timer.prefill():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=1)

    # Prefill already produced one token, hence max_new_tokens - 1.
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and next_token.item() == eos_id:
            break
        with timer.step():
            out = model(next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def main() -> None:
    device = pick_device()
    logger.info(f"device: {device}")

    tokenizer, model = load_model(device=device)
    eos_id = resolve_eos_id(tokenizer)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_tokens = input_ids.shape[1]

    # Correctness: cached decode must be token-for-token identical to naive.
    naive_ids = naive_decode(
        model,
        input_ids,
        max_new_tokens=EQUIVALENCE_TOKENS,
        eos_id=eos_id,
        timer=DecodeTimer(device),
    )
    cached_ids = cached_decode(
        model,
        input_ids,
        max_new_tokens=EQUIVALENCE_TOKENS,
        eos_id=eos_id,
        timer=DecodeTimer(device),
    )
    assert_tokens_match(
        naive_ids[0].tolist(), cached_ids[0].tolist(), candidate_name="phase 1"
    )
    logger.info(
        f"equivalence OK — output: {tokenizer.decode(cached_ids[0], skip_special_tokens=True)!r}"
    )

    # Latency vs sequence length (EOS disabled so every run is exactly `n` tokens).
    for n in GEN_LENGTHS:
        logger.info(f"benchmarking gen={n} (naive O(n^2) — slow at large n)…")
        naive_timer = DecodeTimer(device)
        cached_timer = DecodeTimer(device)
        naive_decode(model, input_ids, max_new_tokens=n, eos_id=None, timer=naive_timer)
        cached_decode(
            model, input_ids, max_new_tokens=n, eos_id=None, timer=cached_timer
        )
        p0 = naive_timer.build(prompt_tokens=prompt_tokens)
        p1 = cached_timer.build(prompt_tokens=prompt_tokens)
        logger.info(
            f"gen={n:<4}  "
            f"phase0 {p0.median_step_ms:6.1f} ms/tok ({p0.decode_tokens_per_s:6.1f} tok/s)   "
            f"phase1 {p1.median_step_ms:6.1f} ms/tok ({p1.decode_tokens_per_s:6.1f} tok/s)   "
            f"{p1.decode_tokens_per_s / p0.decode_tokens_per_s:4.1f}x"
        )


if __name__ == "__main__":
    main()
