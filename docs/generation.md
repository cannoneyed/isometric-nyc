# Generation

The app uses a number of bespoke tools for managing generation. The main tool is a web app that contains a number of utilities that allow you to generate new tiles on demand, using a variety of models/approaches.

<Video src="https://cannoneyed.com/img/projects/isometric-nyc/gen_app_screencast.webm" />

## Setup

First, you need to install the web dependencies:

```bash
cd src/web_render && bun i
```

Then, you can run the app. By default, it will use the `MAP_ID` env variable to determine which generations dir to use.

```bash
uv run python src/isometric_nyc/generation/app.py
```

## Features

### Tools
| Tool | Description |
| ---- | ----------- |
| Select | The select tool allows you to select up to four quadrants at a time for generation or defining a rectangle |
| Fix Water | Toggle a color select eye dropper to pick from a given tile. After color selection, you can click the `Apply Fix` button to do a fill+replace color operation to fix water colors. |
| Fill Water | Fills a tile with water. |
| Water Select | Toggles whether the tile is labeled as water in the db |
| Deselect | Clears the current selection |
| Delete | Deletes the selected tile(s). If two tiles are selected, it can operate as deleting the rectangle selection defined by the  top-left and bottom-right selected quadrants (dangerous!) |
| Flag | Adds a flag to the quadrant |
| Star | Adds a star to the quadrant |
| Starred | Lets you toggle through starred quadrants |
| Render | Generates the 3D satellite render for a selection |
| Generate | Generates the `generations` data (or other layer for the selection) |
| + Prompt | Sets an additional prompt to be appended to the base prompt for the model |
| - Prompt | Sets a negative prompt |
| Gen Rect | Automatically creates a tiled generation plan for the given rectangle defined by two top-left and bottom-right selected quadrants, and adds them to the queue |
| 💧 Fill Rect | Fills the selected rectangle with water |
| Export cmd | Gives a CLI command for exporting/importing the selected rectangle as png. Useful for manual editing in e.g. Affinity |
| Export | Exports the given selected rectangle as a png and downloads |

## Generating

In order to generate tiles in the app, select up to 4 quadrants using the select tool and click "Generate". Generation has a number of rules to prevent seams from forming between tile quadrants:

![a sample of generation rules](https://cannoneyed.com/projects/isometric-nyc/img/projects/isometric-nyc/gen_rules.png)

## 2x2 generation
A 2x2 generation is only legal if the 2x2 tile does not touch any previously generated quadrants

```
XXXXXGG  XXXXX
XXSSXGG  XXSSG
XXSSXGG  XXSSG
XXXXXGG  XXXXX
✅ OK    ⛔ NO

X = empty, S = selected, G = generated
```

## 1x2 or 2x1 generation
A 2x1 or 2x2 generation is legal if the 

```
XXXXXX  XXXXX  
XXGGSX  XGGSG
XXGGSX  XGGSG
XXXXXX  XXXXX
✅ OK   ✅ OK

X = empty, S = selected, G = generated
```

### From Modal

See the [modal](http://modal.com/) inference docs for how to set up a modal inference server for a fine-tuned model. Once you have a model running, set the `MODAL_INFERENCE_URL` env variable to the base64 edit URL

```bash
MODAL_INFERENCE_URL=https://your-workspace--qwen-image-edit-server-imageeditor-edit-b64.modal.run
```

This will allow you to generate tiles using the fine-tuned model from the app.

### From Oxen

You can also use fine-tuned models hosted on [oxen.ai](https://oxen.ai) to generate tiles. First, ensure your model is up and deployed on the oxen application (see [docs](https://docs.oxen.ai/examples/fine-tuning/image_editing#deploying-the-model)).

You then need to set the `model_id` in the `app_config.json` for the model you'd like to use. Finally, get an API key for the model and set the `OXEN_MODEL_API_KEY` env variable.

### Nano Banana

Nano Banana is also available as a generation source - it requires a `GEMINI_API_KEY` env variable and by default uses the prompt that's manually set via the app UI.
