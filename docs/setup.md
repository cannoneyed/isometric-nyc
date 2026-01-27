# Setup

## Environment variables

First, copy the .env file and set the necessary environment variables

```bash
cp .env.example .env
```

## Python

isometric.nyc uses [`uv`](https://docs.astral.sh/uv/) for python dependency management and runtimes. First, ensure `uv` is installed and then run:

```bash
uv sync
```

This will install all python dependencies and set up the necessary python environment.

## Tiles

Tile data is stored in a sqlite db in `generations/<map-name>`. This repo ships with a script to download a small subset of the NYC map at `generations/tiny-nyc`. The default generation directory that will be used is defined by the `MAP_ID` environment variable (defaulting to `tiny-nyc` in the .env file template).

```bash
uv run python src/isometric_nyc/download_generation.py tiny-nyc
```

If you'd like to download the full NYC tile dataset, follow the *Full Dataset* instructions at the bottom of this README.

## Creating the DZI tiles

The web application uses OpenSeaDragon and the DZI tile viewer format. This repo ships with a pre-computed DZI image pyramid, but you can reconstruct it at any time by running the `export_dzi` script:

```bash
uv run python src/isometric_nyc/generation/export_dzi.py
```

## Web application

isometric nyc uses [`bun`](https://bun.com/) to build, manage, and run JavaScript and frontend apps. First, ensure that `bun` is installed.

You can run the isometric nyc web application with the following command:

```
cd src/app
bun i
bun run dev
```

The web app will run and point to the dzi image pyramid at `src/app/public/dzi/<MAP_ID>` (with `MAP_ID` corresponding to the `MAP_ID` .env variable, defaulting to `tiny-nyc`)

See the web application [docs](docs/app.md) for more information.

---

## Full Dataset

In order to download and initialize the full NYC tiles dataset, you'll need to load the data from [oxen.ai](https://oxen.ai).

```bash
cd oxen/datasets
oxen clone https://hub.oxen.ai/cannoneyed/isometric-nyc-tiles
```

Then, you'll need to populate the generations dir with the downloaded png files

```bash
# From the repo root
uv run python src/isometric_nyc/oxen_sync/import_from_oxen.py --generations_dir nyc --oxen_dataset oxen/isometric-nyc-tiles
```

This will populate the `generations/nyc/quadrants.db` with the data and metadata for the generation. You can then create the DZI tiles for the new generations database using the instructions above. Ensure you set the MAP_ID env variable to use the new `nyc` generations dir by default.

```bash
# .env
MAP_ID="nyc"  # The full map generations dir
```

```bash
uv run python src/isometric_nyc/generation/export_dzi.py
```

Then, you can start up the web app with the new local dzi tiles for the full map.

```bash
cd src/app
bun run dev
```
