lint:
  uvx ruff format
  uvx ruff check --fix

typecheck:
  uv run pyright