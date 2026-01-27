import base64
import os
import random
from io import BytesIO

import modal
from pydantic import BaseModel

# Configuration via environment variables
# Set LORA_MODEL_ID when deploying to use a different LoRA
# Set LORA_WEIGHT_NAME to specify a checkpoint file (e.g., "checkpoint-500.safetensors")
DEFAULT_LORA_MODEL_ID = "cannoneyed-dark-copper-flea"
LORA_MODEL_ID = os.environ.get("LORA_MODEL_ID", DEFAULT_LORA_MODEL_ID)
LORA_WEIGHT_NAME = os.environ.get("LORA_WEIGHT_NAME", None)  # None = use default
print(
  f"📦 Deploying with LORA_MODEL_ID={LORA_MODEL_ID}, LORA_WEIGHT_NAME={LORA_WEIGHT_NAME}"
)

# 1. Define the Environment
image = modal.Image.debian_slim(python_version="3.11").pip_install(
  "torch",
  "torchvision",
  "diffusers",
  "transformers",
  "accelerate",
  "peft",
  "pillow",
  "fastapi",
  "uvicorn",
  "python-multipart",
)

# Include short LoRA identifier in app name to ensure separate containers for different models
# Use last part of the model ID (e.g., "amethyst-vulture" from "cannoneyed-dependent-amethyst-vulture")
LORA_SHORT_ID = "-".join(LORA_MODEL_ID.split("-")[-2:]) if "-" in LORA_MODEL_ID else LORA_MODEL_ID
app = modal.App(f"qwen-edit-{LORA_SHORT_ID}")

# 2. Attach the volume containing LoRA adapters
# Volume structure: /data/loras/<model-id>/
# See README.md for upload instructions
lora_volume = modal.Volume.from_name("isometric-lora-vol", create_if_missing=True)


# 3. Define the Request Model
class EditRequest(BaseModel):
  image_b64: str
  prompt: str
  negative_prompt: str | None = None
  true_cfg_scale: float = 2.0  # Default moved here
  steps: int = 14  # Default moved here
  guidance_scale: float = 3.0  # Default moved here
  seed: int | None = None


# 4. The Serverless Class
@app.cls(
  image=image,
  gpu="H100",  # High memory needed for Qwen-Image-Edit
  volumes={"/data": lora_volume},
  scaledown_window=300,  # Keep container alive for 5 mins after last request
  timeout=600,  # Allow 10 mins for cold-start compilation
  secrets=[
    modal.Secret.from_dict(
      {
        "LORA_MODEL_ID": LORA_MODEL_ID,
        "LORA_WEIGHT_NAME": LORA_WEIGHT_NAME or "",
      }
    )
  ],
)
class ImageEditor:
  @modal.enter()
  def setup(self):
    """This runs once when the container boots up."""
    import os

    import torch
    from diffusers import QwenImageEditPipeline

    # Cache torch.compile artifacts to volume for faster subsequent cold starts
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch_cache"

    # Determine LoRA path from environment
    self.lora_model_id = os.environ.get("LORA_MODEL_ID", "cannoneyed-dark-copper-flea")
    self.lora_weight_name = os.environ.get("LORA_WEIGHT_NAME", "") or None
    lora_path = f"/data/loras/{self.lora_model_id}"

    print(f"🚀 Loading Qwen-Image-Edit Pipeline with LoRA: {self.lora_model_id}")
    if self.lora_weight_name:
      print(f"   Using checkpoint: {self.lora_weight_name}")

    # Load Base Pipeline
    self.pipe = QwenImageEditPipeline.from_pretrained(
      "Qwen/Qwen-Image-Edit",
      torch_dtype=torch.bfloat16,
    )

    # Load LoRA from the attached volume
    print(f"🔗 Loading LoRA adapter from {lora_path}...")
    try:
      self.pipe.load_lora_weights(
        lora_path,
        adapter_name="isometric",
        weight_name=self.lora_weight_name,
      )
      self.pipe.set_adapters(["isometric"], adapter_weights=[1.0])
    except Exception as e:
      print(f"⚠️ Failed to load LoRA (check volume path): {e}")

    self.pipe.to("cuda")

    # torch.compile disabled - adds ~2min to cold start for ~10-20% inference speedup
    # Uncomment below to enable:
    # print("🔨 Compiling transformer...")
    # self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")

    print("✅ Model loaded and ready!")

  @modal.fastapi_endpoint(method="POST")
  async def edit_b64(self, req: EditRequest):
    """Base64 JSON Endpoint"""
    from PIL import Image

    # Decode
    image_data = base64.b64decode(req.image_b64)
    input_image = Image.open(BytesIO(image_data)).convert("RGB")

    # Run Inference
    result_img = self._inference(
      input_image,
      req.prompt,
      req.negative_prompt,
      req.true_cfg_scale,
      req.steps,
      req.guidance_scale,
      req.seed,
    )

    # Encode Response
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {"image_b64": result_b64}

  def _inference(self, image, prompt, neg_prompt, true_cfg, steps, guide_scale, seed):
    """Internal helper to avoid code duplication"""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    if seed is None:
      seed = random.randint(0, 2**32 - 1)

    print(f"🎨 Edit: '{prompt}' | Model: {self.lora_model_id} | Seed: {seed}")

    with torch.inference_mode():
      output = self.pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        true_cfg_scale=true_cfg,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guide_scale,
        generator=torch.manual_seed(seed),
      )

    return output.images[0]
