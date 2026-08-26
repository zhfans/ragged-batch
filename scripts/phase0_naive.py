import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = model.to(device)  # pyright: ignore
    model.eval()

    eos_id = tokenizer.eos_token_id

    tokens = tokenizer("Hello, my name is", return_tensors="pt")
    input_ids = tokens.input_ids.to(device)

    with torch.no_grad():
        for _ in range(100):
            logits = model(input_ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                break

    logger.info(
        f"Generated text: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}"
    )


if __name__ == "__main__":
    main()
