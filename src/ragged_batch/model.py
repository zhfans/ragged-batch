"""Model and tokenizer loading, shared so every phase benchmarks the same weights."""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ragged_batch.device import pick_device

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load ``model_id``'s tokenizer and causal-LM weights on ``device``, in eval mode.

    ``device`` defaults to :func:`ragged_batch.device.pick_device`. Weights
    (~1 GB for the default model) download and cache from the Hugging Face Hub on
    first use.
    """
    if device is None:
        device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)  # pyright: ignore[reportArgumentType] -- PreTrainedModel.to() is `@wraps(nn.Module.to)`, which pyright misresolves against nn.Module.__call__
    model.eval()

    return tokenizer, model


def resolve_eos_id(tokenizer: PreTrainedTokenizerBase) -> int | None:
    """Narrow ``tokenizer.eos_token_id`` to a single int, or ``None``.

    ``PreTrainedTokenizerBase`` types the attribute as a broad union: a
    tokenizer may in principle carry several EOS tokens as a list, or none.
    Every model these phases run has exactly one integer EOS, so collapse the
    union here and let the decode loops take a plain ``int | None``.
    """
    eos = tokenizer.eos_token_id
    return eos if isinstance(eos, int) else None
