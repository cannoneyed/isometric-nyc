# Isometric NYC Oxen Data

This directory contains the oxen data for the isometric nyc tiles. The NYC tiles (actually known as quadrants) are maintained at [https://www.oxen.ai/cannoneyed/isometric-nyc-tiles](https://www.oxen.ai/cannoneyed/isometric-nyc-tiles).

## Getting started

To fetch the tiles, first ensure that `oxen` is installed ([docs](https://docs.oxen.ai/getting-started/install))

```
brew install oxen   # MacOS
```

Next, clone the repo in this directory. The repo is ~10GB so this may take a long time!

```
cd oxen  # From the root directory
oxen clone https://hub.oxen.ai/cannoneyed/isometric-nyc-tiles
```

Once the raw tile quadrants are loaded, you can load them into the generations db to develop/build locally.

```
# From the root directory
uv run python src/isometric_nyc/oxen_sync/import_from_oxen.py
```

This will populate a new SQLite generations DB with the tile quadrant data in `generations/nyc`

## Pushing changes

If you've made any changes to the tile generation data, you can export those changes and then push those changes to a new branch on oxen. Note, you'll need to fork the public repo and make your own in order to submit a "PR" against the data.

```
# From the root directory
uv run python src/isometric_nyc/oxen_sync/export_from_oxen.py --oxen_repo <repo-name>

cd oxen/<repo-name>
oxen add .
oxen commit --message "<your commit message>"
oxen push
```
