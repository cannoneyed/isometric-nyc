## Models

The model that generated the vast majority of the tile data was a fine-tuned `Qwen/Image-Edit` model trained on [oxen.ai]. These models were trained on an "omni" infill task - essentially they were trained to generate pixel-art-style data in an image with some portion of previously-generate pixel data "masked" with a rectangle of raw, orthographically rendered 3D tiles satellite data (see examples below).

![infill pair images / diagram](https://cannoneyed.com/img/projects/isometric-nyc/training_data_infill.png)

These synthetic datasets were assembled from render->generation pairs generated with [Nano Banana](https://gemini.google/overview/image-generation/), using a [marimo](https://marimo.io) notebook in `src/notebooks/nano-banana.py`.

There were a few (poorly named) dataset / model combinations that accounted for the vast majority of the generated tiles in the application:

| **oxen repo** | **model** | **data** | **about** |
|---|---|---|--|
|[cannoneyed/isometric-nyc](https://www.oxen.ai/cannoneyed/isometric-nyc)| [`cannoneyed-rural-rose-dingo`](https://www.oxen.ai/cannoneyed/isometric-nyc/fine-tunes/720944c1-89f5-42f8-80cf-3730c62dec5c) | [omni_v04.csv](https://www.oxen.ai/cannoneyed/isometric-nyc/file/68f95b8e1108f15476220191d6df0d5c/omni_v04.csv)| Omni infill model w/ water tiles |
|[cannoneyed/isometric-nyc](https://www.oxen.ai/cannoneyed/isometric-nyc)| [`cannoneyed-quiet-green-lamprey`](https://www.oxen.ai/cannoneyed/isometric-nyc/fine-tunes/b32cb472-2874-45fb-ae9f-8cecd06174db) | [omni_v04.csv](https://www.oxen.ai/cannoneyed/isometric-nyc/file/666bb2b7ee3d14a166f828a7ce176587/omni_v04.csv)| Omni infill model w/ more trees
|[cannoneyed/isometric-nyc-v2](https://www.oxen.ai/cannoneyed/isometric-nyc-v2)| [`cannoneyed-dark-copper-flea`](https://www.oxen.ai/cannoneyed/isometric-nyc-v02/fine-tunes/354ec4f4-f612-4add-b129-83e02e3803bf)| [omni.csv](https://www.oxen.ai/cannoneyed/isometric-nyc-v02/file/c39b410e3e2e63768a85117ac3324bcb/omni.csv) |v2 omni infill w/ more terrain|

## Downloading weights / running inference

See [inference/README.md](inference/README.md)
