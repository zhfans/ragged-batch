"""Phase 0 — naive autoregressive decoding.

Greedy decode by raw forward pass with no KV cache: every step re-processes the
whole sequence from position 0 (O(n) work per step, O(n^2) over a generation).
Slow but trivially correct — the token-for-token oracle every later phase must
reproduce.
"""

import torch
from loguru import logger
from transformers import PreTrainedModel

from ragged_batch import DecodeTimer, load_model, pick_device, resolve_eos_id

PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 100


@torch.no_grad()
def naive_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None,
    timer: DecodeTimer,
) -> torch.Tensor:
    """Greedy-decode from ``input_ids`` with no cache; return prompt + generated ids.

    Every step feeds the entire running sequence back through the model
    (``use_cache=False``) and keeps only the last position's logits. The first
    pass (prompt only) is timed as prefill so its cost lines up with the cached
    phase's prefill; the rest are decode steps.
    """
    generated = input_ids
    for step in range(max_new_tokens):
        region = timer.prefill() if step == 0 else timer.step()
        with region:
            logits = model(generated, use_cache=False).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_id is not None and next_token.item() == eos_id:
            break
    return generated


def main() -> None:
    device = pick_device()
    logger.info(f"device: {device}")

    tokenizer, model = load_model(device=device)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    timer = DecodeTimer(device)
    generated = naive_decode(
        model,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_id=resolve_eos_id(tokenizer),
        timer=timer,
    )

    logger.info(f"output: {tokenizer.decode(generated[0], skip_special_tokens=True)!r}")
    logger.info(timer.build(prompt_tokens=input_ids.shape[1]).summary())


if __name__ == "__main__":
    main()
