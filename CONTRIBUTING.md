# Contributing to Isometric NYC

Thank you for your interest in contributing to Isometric NYC!

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies using `uv sync`
4. Create a new branch for your feature or fix

## Development Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run tests (not many...)
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the required API keys:

```bash
cp .env.example .env
```

## Code Style

- Use Python type hints for all function signatures
- Use absolute imports for project modules
- Follow the existing code style (enforced by ruff)
- Keep code simple and focused

## Submitting Changes

1. Ensure all tests pass: `uv run pytest`
2. Ensure code is formatted: `uv run ruff format .`
3. Ensure code passes linting: `uv run ruff check .`
4. Commit your changes with a clear message
5. Push to your fork and submit a pull request

## Questions?

Feel free to open an issue if you have questions or need help.
