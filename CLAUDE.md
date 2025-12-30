# Isometric NYC

This repo contains the code for generating an isometric pixel art view of New York City using the latest and greatest AI tools available.

## Technology Stack

  * **Language:** Python (Version specified in `.python-version` or `pyproject.toml`)
  * **Package Manager:** `uv` (Strictly used for all dependency management and environment handling)
  * **Testing:** `pytest`
  * **Linting/Formatting:** `ruff`

## Environment & Dependency Management (`uv`)

**Crucial:** Do not use `pip`, `poetry`, or `conda`. This project relies entirely on `uv`.

### Setup

To initialize the environment and install dependencies:

```bash
uv sync
```

*This creates the virtual environment automatically in `.venv`.*

### Managing Dependencies

  * **Add a package:** `uv add <package_name>`
  * **Add a dev dependency:** `uv add --dev <package_name>`
  * **Remove a package:** `uv remove <package_name>`
  * **Update lockfile:** `uv lock`

### Running Code

Always execute commands within the project's context using `uv run`. Do not manually activate the virtual environment.

  * **Run a script:** `uv run python src/main.py`
  * **Run a module:** `uv run python -m my_module`
  * **Run arbitrary tools:** `uv run <command>`

## Common Tasks & Commands

| Task | Command |
| :--- | :--- |
| **Run Tests** | `uv run pytest` |
| **Format Code** | `uv run ruff format .` (Adjust if using different tool) |
| **Lint Code** | `uv run ruff check .` (Adjust if using different tool) |
| **Run Application** | `uv run python src/app.py` |

## Directory Structure

```text
.
├── .venv/               # Managed by uv (do not touch)
├── src/                 # Source code
│   └── isometric_nyc/   # Main package
├── tests/               # Test suite
├── tasks/               # Task files, which are markdown files that define new
│                        # tasks for the agent to execute (see below)
├── pyproject.toml       # Project configuration & dependencies
├── uv.lock              # Exact dependency versions (do not edit manually)
├── .python-version      # Target Python version
└── README.md            # Human-readable documentation
```

## Coding Standards

1.  **Type Hints:** All function signatures must include Python type hints.
2.  **Imports:** Use absolute imports for project modules (e.g., `from isometric_nyc.utils import helper`).
3.  **Configuration:** All project metadata and tool configuration must reside in `pyproject.toml`.
4.  **Lockfile:** Never ignore `uv.lock`. If `pyproject.toml` changes, run `uv lock` or `uv sync` to update it.

## Task Files

Task files are markdown files that define new tasks for the agent to execute. They are located in the `tasks` directory. These files may be edited by the agent to clarify instructions or to add new tasks, but the use must specify invoking any specific task.

-----
