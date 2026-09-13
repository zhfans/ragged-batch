"""Phase 2 — static batching.

Left-pad several prompts to one common length and decode them together: one
shared input tensor, one shared KV cache with a batch axis, one forward pass per
step covering every sequence at once. Two things fall out of that:

Every row must keep decoding every step, even after it's "done" — a padded
batch tensor can't drop a finished row without reshaping itself, so a short
request rides along, unused, until the batch's longest request finishes. That
forced ride-along *is* the head-of-line blocking this phase exists to surface;
Phase 3's per-sequence cache is the fix. Requests finish at different times for
two real reasons: the model emits EOS, or the client only asked for so many
tokens. Qwen2.5-0.5B is a base (not instruction-tuned) model and rarely emits
EOS on short free-form continuations, so this script models the second reason
directly via a per-request token budget — a more reliable way to demonstrate
the same real-world scenario than hoping for an EOS that may not come.

Left-padding also means position 0 isn't column 0 anymore: ``position_ids``
must be derived from ``attention_mask`` (real tokens count from 0; padding is
clamped and simply never attended to) and passed explicitly on every forward
call, prefill and decode alike — the model would otherwise default to
``0..seq_len-1`` and silently shift every padded row's RoPE positions.

Unlike Phase 0 vs Phase 1, this script doesn't assert batched-vs-solo token
equivalence. Padding widens the attention/matmul shapes, which changes
floating-point reduction order, so a near-tied argmax can legitimately land on
either side of that noise — but it wouldn't matter here either way: neither
number this phase reports (wasted steps, speedup) depends on token content,
only on step counts, and step counts are driven entirely by each row's token
budget, not by what the model actually predicts. What's checked instead is the
one thing those numbers *do* depend on: that ``finished_step`` lands where the
budget says it should.
"""

import torch
from loguru import logger
from phase1_kv_cache import cached_decode
from transformers import PreTrainedModel

from ragged_batch import DecodeTimer, load_model, pick_device, resolve_eos_id

# (prompt, requested output length) -- the spread in requested lengths is what
# makes head-of-line blocking visible: the batch can't finish faster than its
# longest request.
REQUESTS = [
    ("Hello, my name is", 10),
    ("The capital of France is a city called", 40),
    ("The weather today is", 15),
    ("In the year 2050,", 25),
]


@torch.no_grad()
def static_batch_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: torch.Tensor,
    eos_id: int | None,
    timer: DecodeTimer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy-decode a left-padded batch together; return ``(generated, finished_step)``.

    ``max_new_tokens`` is a per-row token budget (shape ``[batch]``). A row is
    "finished" once it hits EOS *or* exhausts its own budget, whichever comes
    first; the batch keeps running every row until all are finished. That's why
    ``finished_step`` -- the 0-indexed generation step at which each row first
    finished (step 0 is prefill's own token) -- is never ``-1``: every row is
    guaranteed to finish by its own budget at the latest.

    ``generated`` is ``[batch, prompt_width + steps_run]``, still left-padded on
    the prompt side.
    """
    batch_size, device = input_ids.shape[0], input_ids.device
    steps_cap = int(max_new_tokens.max().item())
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    with timer.prefill():
        out = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
    cache = out.past_key_values
    next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_tokens], dim=1)

    def _eos_hit() -> torch.Tensor:
        if eos_id is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return next_tokens.squeeze(-1) == eos_id

    finished = _eos_hit() | (max_new_tokens <= 1)
    finished_step = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    finished_step[finished] = 0

    cur_mask = attention_mask
    cur_position_ids = position_ids[:, -1:] + 1
    for step in range(1, steps_cap):
        if bool(finished.all()):
            break
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_tokens)], dim=1)
        with timer.step():
            out = model(
                next_tokens,
                attention_mask=cur_mask,
                position_ids=cur_position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        cache = out.past_key_values
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)
        newly_finished = (_eos_hit() | (max_new_tokens <= step + 1)) & ~finished
        finished_step[newly_finished] = step
        finished = finished | newly_finished
        cur_position_ids = cur_position_ids + 1

    return generated, finished_step


def main() -> None:
    device = pick_device()
    logger.info(f"device: {device}")

    tokenizer, model = load_model(device=device)
    eos_id = resolve_eos_id(tokenizer)
    tokenizer.padding_side = "left"

    prompts = [prompt for prompt, _ in REQUESTS]
    budgets = torch.tensor([budget for _, budget in REQUESTS], device=device)

    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    timer = DecodeTimer(device)
    generated, finished_step = static_batch_decode(
        model,
        input_ids,
        attention_mask,
        max_new_tokens=budgets,
        eos_id=eos_id,
        timer=timer,
    )

    sequential_s = 0.0
    for i, (prompt, budget) in enumerate(REQUESTS):
        # The number the report below actually depends on: this row must finish
        # exactly at its own budget. Failing here means the budget/EOS bookkeeping
        # is wrong -- e.g. an off-by-one in `finished_step` -- not that the batched
        # and solo runs said different words.
        finished = int(finished_step[i])
        assert finished == budget - 1, (
            f"row {i} finished at step {finished}, expected {budget - 1} "
            f"({prompt!r}, budget={budget}) -- or EOS fired before the budget did"
        )

        solo_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        solo_timer = DecodeTimer(device)
        cached_decode(
            model, solo_ids, max_new_tokens=budget, eos_id=eos_id, timer=solo_timer
        )
        sequential_s += solo_timer.build(prompt_tokens=solo_ids.shape[1]).total_s
    logger.info("budget check OK — every row finished exactly at its own budget")

    total_generated = generated.shape[1] - input_ids.shape[1]
    logger.info(f"head-of-line blocking (batch ran {total_generated} steps total):")
    for i, (prompt, budget) in enumerate(REQUESTS):
        wasted = (total_generated - 1) - int(finished_step[i])
        logger.info(
            f"  [{i}] wanted {budget:>2} tokens, {wasted:>2} wasted step(s)  {prompt!r}"
        )

    batched_trace = timer.build(prompt_tokens=input_ids.shape[1])
    batched_tokens_per_s = (
        len(REQUESTS) * batched_trace.decode_steps / batched_trace.decode_s
    )
    logger.info(
        f"batched    {batched_trace.total_s * 1e3:7.1f} ms total  "
        f"({batched_trace.decode_steps} steps, {batched_tokens_per_s:6.1f} tok/s aggregate)"
    )
    logger.info(
        f"sequential {sequential_s * 1e3:7.1f} ms total  (phase 1, one prompt at a time)"
    )
    logger.info(f"speedup: {sequential_s / batched_trace.total_s:.2f}x")


if __name__ == "__main__":
    main()
