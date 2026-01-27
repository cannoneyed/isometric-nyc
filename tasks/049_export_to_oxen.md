# Import / Export to/from oxen

I want to set up a way to import/export the data from my generations db into an oxen.ai dataset VCS. Please research the oxen.ai docs and figure out a way to implement something like the following:

`uv run export_to_oxen --generations_dir <default=v01> --oxen_dataset <oxen_dataset_id>`

This command should take all of the tiles from generations db and save them in an oxen database using the following structure:

```
renders/
  <xxx>_<yyy>_<hash>.png
quadrants/
  <xxx>_<yyy>_<hash>.png
README.md
generations.csv  # CSV dataset with all of the generated tiles, in the format `render,quadrant,x,y`
```

All quadrants and renders will be exported into the dataset. If the hash is different, we replace the render/generation. If it's the same, we ignore it and don't update/change the data.

We also want to expose the following:

`uv run export_to_oxen --generations_dir <default=v01> --oxen_dataset <oxen_dataset_id>`

which is the exact inverse of the export - given the oxen dataset and the data, import all of it into the SQLite db.

Please check all of the logic and come up with an implementation plan below:

## Implementation Plan

### Overview

Create two CLI commands for bidirectional sync between the local SQLite generations database and an oxen.ai remote dataset. The hash-based approach ensures efficient incremental updates.

### Dependencies

Add `oxenai` package:
```bash
uv add oxenai
```

### File Structure

Create new module at `src/isometric_nyc/oxen_sync/`:
```
src/isometric_nyc/oxen_sync/
├── __init__.py
├── export_to_oxen.py    # Export command
├── import_from_oxen.py  # Import command
└── utils.py             # Shared utilities (hashing, file naming)
```

### Hash Strategy

The hash will be computed from the **PNG blob content** using MD5 (first 8 characters):
```python
import hashlib

def compute_hash(blob: bytes) -> str:
    """Compute short hash for image content."""
    return hashlib.md5(blob).hexdigest()[:8]
```

File naming: `{quadrant_x:04d}_{quadrant_y:04d}_{hash}.png`
- Example: `0012_0045_a1b2c3d4.png`

### Data Model

**generations.csv schema:**
```csv
render,quadrant,x,y,hash_render,hash_quadrant
renders/0012_0045_a1b2c3d4.png,quadrants/0012_0045_e5f6g7h8.png,12,45,a1b2c3d4,e5f6g7h8
```

### Export Command (`export_to_oxen.py`)

**CLI Interface:**
```bash
uv run python -m isometric_nyc.oxen_sync.export_to_oxen \
    --generations_dir v01 \
    --oxen_dataset namespace/dataset-name \
    [--branch main]
```

**Algorithm:**
1. **Authentication**: Load oxen auth from environment or config
2. **Connect to remote**: Use `RemoteRepo` and `Workspace` API
3. **Load existing state**:
   - Download `generations.csv` if it exists
   - Build a lookup dict: `{(x, y): (render_hash, quadrant_hash)}`
4. **Iterate local quadrants**:
   - Query SQLite for all quadrants with `render IS NOT NULL` or `generation IS NOT NULL`
   - For each quadrant:
     - Compute hash of render blob (if exists)
     - Compute hash of generation blob (if exists)
     - Compare with remote hashes
     - If different or new: add to upload queue
5. **Upload changed files**:
   - Write blobs to temp files with correct naming
   - Use `workspace.add()` to stage files
   - Batch uploads to avoid memory issues
6. **Update CSV**:
   - Regenerate `generations.csv` with all current mappings
   - Use `workspace.add()` to stage CSV
7. **Generate README.md**:
   - Basic info about dataset, generation date, quadrant count
8. **Commit and push**:
   - `workspace.commit("Sync from local generations db")`

**Pseudocode:**
```python
def export_to_oxen(generations_dir: str, oxen_dataset: str, branch: str = "main"):
    db_path = Path(f"generations/{generations_dir}/quadrants.db")

    # Connect to oxen
    repo = RemoteRepo(oxen_dataset)
    workspace = Workspace(repo, branch)

    # Load existing CSV (if any)
    existing = load_existing_csv(workspace)  # Dict[(x,y)] -> (render_hash, gen_hash)

    # Query all quadrants from SQLite
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT quadrant_x, quadrant_y, render, generation
        FROM quadrants
        WHERE render IS NOT NULL OR generation IS NOT NULL
    """).fetchall()

    new_csv_rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for x, y, render_blob, gen_blob in rows:
            render_hash = compute_hash(render_blob) if render_blob else None
            gen_hash = compute_hash(gen_blob) if gen_blob else None

            existing_render_hash, existing_gen_hash = existing.get((x, y), (None, None))

            # Export render if changed
            if render_blob and render_hash != existing_render_hash:
                render_path = f"renders/{x:04d}_{y:04d}_{render_hash}.png"
                write_temp_file(tmpdir, render_path, render_blob)
                workspace.add(f"{tmpdir}/{render_path}", "renders/")

            # Export generation if changed
            if gen_blob and gen_hash != existing_gen_hash:
                gen_path = f"quadrants/{x:04d}_{y:04d}_{gen_hash}.png"
                write_temp_file(tmpdir, gen_path, gen_blob)
                workspace.add(f"{tmpdir}/{gen_path}", "quadrants/")

            # Build CSV row
            new_csv_rows.append({
                "x": x, "y": y,
                "render": f"renders/{x:04d}_{y:04d}_{render_hash}.png" if render_hash else "",
                "quadrant": f"quadrants/{x:04d}_{y:04d}_{gen_hash}.png" if gen_hash else "",
                "hash_render": render_hash or "",
                "hash_quadrant": gen_hash or ""
            })

        # Write and upload CSV
        write_csv(f"{tmpdir}/generations.csv", new_csv_rows)
        workspace.add(f"{tmpdir}/generations.csv")

        # Generate README
        write_readme(f"{tmpdir}/README.md", len(new_csv_rows))
        workspace.add(f"{tmpdir}/README.md")

        # Commit
        workspace.commit(f"Export {len(new_csv_rows)} quadrants from {generations_dir}")
```

### Import Command (`import_from_oxen.py`)

**CLI Interface:**
```bash
uv run python -m isometric_nyc.oxen_sync.import_from_oxen \
    --generations_dir v01 \
    --oxen_dataset namespace/dataset-name \
    [--branch main]
```

**Algorithm:**
1. **Connect to remote**: Use `RemoteRepo` API
2. **Download generations.csv**: Parse to get all file mappings
3. **Compute local hashes**:
   - Query SQLite for all quadrants
   - Compute hashes of existing blobs
4. **Compare and download**:
   - For each row in CSV where hash differs from local:
     - Download render PNG (if hash differs)
     - Download quadrant PNG (if hash differs)
5. **Update SQLite**:
   - Use `INSERT OR REPLACE` to update quadrant blobs
6. **Report summary**: Files updated, skipped, errors

**Pseudocode:**
```python
def import_from_oxen(generations_dir: str, oxen_dataset: str, branch: str = "main"):
    db_path = Path(f"generations/{generations_dir}/quadrants.db")

    # Download CSV
    download(repo_id=oxen_dataset, path="generations.csv", dst=tmpdir)
    csv_data = parse_csv(f"{tmpdir}/generations.csv")

    # Load local hashes
    conn = sqlite3.connect(db_path)
    local_hashes = {}
    for row in conn.execute("SELECT quadrant_x, quadrant_y, render, generation FROM quadrants"):
        x, y, render, gen = row
        local_hashes[(x, y)] = (
            compute_hash(render) if render else None,
            compute_hash(gen) if gen else None
        )

    # Download and update changed files
    for row in csv_data:
        x, y = row["x"], row["y"]
        remote_render_hash = row["hash_render"]
        remote_gen_hash = row["hash_quadrant"]
        local_render_hash, local_gen_hash = local_hashes.get((x, y), (None, None))

        updates = {}

        if remote_render_hash and remote_render_hash != local_render_hash:
            download(repo_id=oxen_dataset, path=row["render"], dst=tmpdir)
            updates["render"] = read_blob(f"{tmpdir}/{row['render']}")

        if remote_gen_hash and remote_gen_hash != local_gen_hash:
            download(repo_id=oxen_dataset, path=row["quadrant"], dst=tmpdir)
            updates["generation"] = read_blob(f"{tmpdir}/{row['quadrant']}")

        if updates:
            update_quadrant(conn, x, y, updates)

    conn.commit()
```

### Entry Points (pyproject.toml)

Add script entry points:
```toml
[project.scripts]
export_to_oxen = "isometric_nyc.oxen_sync.export_to_oxen:main"
import_from_oxen = "isometric_nyc.oxen_sync.import_from_oxen:main"
```

### Implementation Steps

1. **Add oxenai dependency**: `uv add oxenai`
2. **Create module structure**: Create `oxen_sync/` directory and files
3. **Implement utils.py**:
   - `compute_hash(blob: bytes) -> str`
   - `format_filename(x: int, y: int, hash: str) -> str`
   - CSV read/write helpers
4. **Implement export_to_oxen.py**:
   - CLI argument parsing (argparse)
   - Database connection and query
   - Oxen workspace operations
   - Progress reporting (tqdm)
5. **Implement import_from_oxen.py**:
   - CLI argument parsing
   - CSV download and parsing
   - File download with hash comparison
   - Database updates
6. **Add tests**:
   - Unit tests for hash computation
   - Integration tests with mock oxen repository
7. **Update pyproject.toml**: Add entry points

### Edge Cases to Handle

- **Empty database**: Export creates empty dataset with just README
- **Missing columns**: Gracefully handle quadrants without render/generation
- **Network errors**: Retry logic for uploads/downloads
- **Large datasets**: Batch uploads to avoid memory issues
- **Existing files with same hash**: Skip upload (no-op)
- **Files in remote not in local**: Don't delete on export (additive only)
- **Quadrant exists in remote but not local**: Create new quadrant row on import

### Configuration

Environment variables:
- `OXEN_API_TOKEN`: Authentication token for oxen.ai
- `OXEN_USER_NAME`: Optional user name for commits
- `OXEN_USER_EMAIL`: Optional email for commits

### Example Usage

```bash
# Initial export
uv run export_to_oxen --generations_dir v01 --oxen_dataset andycoenen/isometric-nyc-tiles

# After generating more tiles, sync again (only uploads changes)
uv run export_to_oxen --generations_dir v01 --oxen_dataset andycoenen/isometric-nyc-tiles

# Import to a fresh machine
uv run import_from_oxen --generations_dir v01 --oxen_dataset andycoenen/isometric-nyc-tiles
```
