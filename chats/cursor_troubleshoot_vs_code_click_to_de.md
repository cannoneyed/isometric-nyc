# Troubleshoot VS Code click-to-definition
_Exported on 12/10/2025 at 09:09:52 EST from Cursor (2.1.39)_

---

**User**

Why isn't click-to-definition working for this vscode python project? e.g. when I try to cmd click on a symbol in @src/isometric_nyc/e2e_generation/generate_tile_omni.py nothing happens

---

**Cursor**

This is likely a Python language server configuration issue. Let me check your project configuration to diagnose the problem.

I found the issue! Your project uses a `src/` layout (where the `isometric_nyc` package lives in `src/isometric_nyc/`), but the Python language server (Pylance) doesn't know to look in the `src/` directory.

You need to add `python.analysis.extraPaths` to your VSCode settings. Let me fix that:

```jsonc
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "python.analysis.extraPaths": ["src"],
  "editor.wordWrap": "wordWrapColumn",
  "editor.wordWrapColumn": 80,
  "editor.rulers": [80]
}

```

I've added `"python.analysis.extraPaths": ["src"]` to your VSCode settings. This tells Pylance (the Python language server) to look in the `src/` directory when resolving imports.

**Why this was needed:** Your project uses a `src/` layout where your package lives at `src/isometric_nyc/`. Without this setting, Pylance doesn't know to look there, so imports like `from isometric_nyc.e2e_generation.generate_tile import generate_tile` can't be resolved.

**To apply the fix:** You'll likely need to reload the window. Press `Cmd+Shift+P` and run "Developer: Reload Window" (or just close and reopen Cursor).

After reloading, `Cmd+click` should work for navigating to definitions.

