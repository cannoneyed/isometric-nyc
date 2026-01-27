# Clean up git repo

I want to deploy the app (and associated code) to github - but first, we need to clean up some of the contents.

Unfortunatley, we seem to have committed at various times many large png files, particularly in the `generations` dir... I want to scrub the git history of those png files.

The only png files we should keep are those in `references`.

I also want to scrub the git history of any `node_modules` that may have been committed.

## ✅ Completed

Used `git-filter-repo` to scrub git history:

1. **Removed all PNG files except `references/`** using regex filter:
   ```bash
   git filter-repo --path-regex '^(?!references/).*\.png$' --invert-paths --force
   ```

2. **Removed all `node_modules/` directories**:
   ```bash
   git filter-repo --path-glob '**/node_modules/**' --invert-paths --force
   ```

### Results
- **Repo size reduced: 1.9GB → 41MB** (~98% reduction)
- **Preserved PNG files**: 7 images in `references/` kept intact
- **node_modules completely removed** from history

### Notes
- A backup branch `backup-before-cleanup-YYYYMMDDHHMMSS` was created before the operation
- The `origin` remote was removed by `git-filter-repo` (standard behavior) - you'll need to re-add it:
  ```bash
  git remote add origin https://github.com/cannoneyed/isometric-nyc.git
  ```
- Since history was rewritten, you'll need to force push:
  ```bash
  git push --force origin <branch-name>
  ```
