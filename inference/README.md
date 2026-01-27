# Inference on Modal

Ensure you have a `modal` account, and that you've initialized:

```bash
uv run modal setup
```

## 1. Download the model weights

Download LoRA adapter weights from Oxen. Each model has a revision hash and model ID.

**Default model (cannoneyed-dark-copper-flea):**

```bash
mkdir -p lora-weights && cd lora-weights
oxen download --revision 55b9eaff9e93ceae2549c85ba2f10b4f cannoneyed/isometric-nyc-v02 models/cannoneyed-dark-copper-flea
cd ..
```

**Additional models:**

```bash
cd lora-weights
oxen download --revision <revision-hash> <oxen-repo-name> models/<model-id>
cd ..
```

Replace `<revision-hash>` and `<model-id>` with values from your Oxen repository.

This creates: `lora-weights/models/<model-id>/`

## 2. Create the Modal volume

```bash
uv run modal volume create isometric-lora-vol
```

## 3. Upload the adapter to the volume

Upload LoRA weights to the volume under `/loras/<model-id>/`:

```bash
# Upload the default model
uv run modal volume put isometric-lora-vol \
  lora-weights/models/cannoneyed-dark-copper-flea \
  /loras/cannoneyed-dark-copper-flea

# Upload additional models (optional)
uv run modal volume put isometric-lora-vol \
  lora-weights/models/another-model-id \
  /loras/another-model-id
```

The server mounts this volume to `/data`, so paths inside the container are `/data/loras/<model-id>/`.

## 4. Run the inference server

For development (hot-reloads on code changes):

```bash
uv run modal serve inference/server.py
```

For production deployment:

```bash
uv run modal deploy inference/server.py
```

To deploy with a specific LoRA model:

```bash
LORA_MODEL_ID=another-model-id uv run modal deploy inference/server.py
```

If not specified, defaults to `cannoneyed-dark-copper-flea`.

Both commands print the endpoint URLs. Look for lines like:

```
└── 🔗 https://your-workspace--qwen-image-edit-server-imageeditor-edit-b64.modal.run
```

## 5. Call the edit_b64 endpoint

The `edit_b64` endpoint accepts JSON with a base64-encoded image and returns a base64-encoded result.

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_b64` | string | yes | - | Base64-encoded input image |
| `prompt` | string | yes | - | Edit instruction |
| `negative_prompt` | string | no | null | What to avoid |
| `true_cfg_scale` | float | no | 2.0 | True CFG scale |
| `steps` | int | no | 14 | Inference steps |
| `guidance_scale` | float | no | 3.0 | Guidance scale |
| `seed` | int | no | random | Seed for reproducibility |

**Python example:**

```python
import base64
import httpx

ENDPOINT = "https://your-workspace--qwen-image-edit-server-imageeditor-edit-b64.modal.run"

# Encode input image
with open("input.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Make request
response = httpx.post(
    ENDPOINT,
    json={
        "image_b64": image_b64,
        "prompt": "convert to isometric pixel art",
    },
    timeout=120,
)

# Save result
result = response.json()
with open("output.png", "wb") as f:
    f.write(base64.b64decode(result["image_b64"]))
```

## 6. Test with quadrant data

Use `test_server.py` to build a template from quadrants in your generation database and call the endpoint:

```bash
uv run python inference/test_server.py \
  --generation-dir generations/nyc \
  --quadrants "(0,0),(1,0),(0,1),(1,1)" \
  --source-layer renders \
  --prompt "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)." \
  --endpoint https://your-workspace--qwen-image-edit-server-imageeditor-edit-b64.modal.run
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--generation-dir` | yes | Path to generation directory containing `quadrants.db` |
| `--quadrants` | yes | Quadrant coordinates to infill, e.g., `"(0,0),(1,0)"` |
| `--source-layer` | no | `renders` or `generations` for context (default: `generations`) |
| `--endpoint` | yes | Modal endpoint URL |
| `--steps` | no | Inference steps (default: 14) |

This saves `input.png` (template) and `output.png` (result) to the `inference/` directory.
