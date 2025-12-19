# Qwen server on localhost
_Exported on 12/19/2025 at 14:46:53 EST from Cursor (2.2.20)_

---

**User**

Run @isometric-nyc-qwen-server/server.py on localhost port 8080

---

**Cursor**

The project already has dependencies set up with `uv`. Let me run the server:

It looks like you skipped the command. Would you like me to run it differently? Some alternatives:

1. **Using uvicorn directly**: `uv run uvicorn server:app --host 0.0.0.0 --port 8080`
2. **Using plain Python** (if dependencies are already installed globally): `python server.py`

Let me know how you'd like to proceed!

---

**User**

no i'm running it directly, what's the advantage of using uvicorn?

---

**Cursor**

```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Good question! In your case, **there's essentially no difference** since your `server.py` already calls uvicorn programmatically:

```86:87:isometric-nyc-qwen-server/server.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Running `python server.py` uses uvicorn under the hood anyway.

**When running uvicorn directly matters:**

| Feature | `python server.py` | `uvicorn server:app --flag` |
|---------|-------------------|----------------------------|
| Hot reload (`--reload`) | ❌ Need to add to code | ✅ Just add flag |
| Multiple workers | ❌ Need to add to code | ✅ `--workers 4` |
| Change port on the fly | ❌ Edit code | ✅ `--port 9000` |

For your ML model server, **none of these matter**:
- Hot reload would reload the model on every code change (slow/wasteful)
- Multiple workers don't help with a single GPU
- Port is already hardcoded to 8080

So `python server.py` is perfectly fine here. 👍

```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

**User**

ok i just called the api - but it's taking a long time to do torch.compile - how long will this take?

---

**Cursor**

`torch.compile` with `mode="reduce-overhead"` on a large diffusion transformer like Qwen-Image-Edit can take **5-15 minutes** on the first inference. It's tracing every operation and generating optimized CUDA kernels.

You'll see it working through the 25 inference steps very slowly as it compiles each unique operation path.

**The good news:** After this first call completes, subsequent requests will be **20-40% faster** since the compiled kernels are cached in memory.

**If you want faster startup for testing**, you could temporarily disable compilation by commenting out this line in `server.py`:

```python
# pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
```

Then re-enable it once you're done iterating. Without compile, first inference will be ~30 seconds instead of 5-15 minutes, but every inference will be slower overall.

```python
# pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
```

---

**User**

I'm noticing that the request still takes ~1:30 to generate. Actual steps are very fast, how can I load the image and start inference faster?

---

**Cursor**

```python
# 3. Use model CPU offload (faster than sequential, keeps submodules on GPU)
pipeline.enable_model_cpu_offload()
```

The culprit is almost certainly `enable_model_cpu_offload()`. It shuttles model components between CPU↔GPU for *every* inference call, which adds massive latency even though the actual compute is fast.

**If you have enough VRAM**, replace CPU offload with keeping the model on GPU:

```python
import torch
import uvicorn
import gc
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from io import BytesIO
from PIL import Image

# CORRECT LIBRARY: Use Diffusers, not Transformers
from diffusers import QwenImageEditPipeline 
from diffusers import FlowMatchEulerDiscreteScheduler

app = FastAPI()

# Global variables
pipeline = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit" 
ADAPTER_PATH = "/home/ubuntu/isometric-nyc-utah/models/cannoneyed-rural-rose-dingo"

@app.on_event("startup")
async def load_model():
    global pipeline
    print("🚀 Loading Qwen-Image-Edit Pipeline...")
    
    # 1. Load the Base Pipeline in bfloat16 (no quantization - causes issues with LoRA)
    pipeline = QwenImageEditPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # 2. Load Your Oxen LoRA
    print(f"🔗 Loading LoRA adapter from {ADAPTER_PATH}...")
    pipeline.load_lora_weights(ADAPTER_PATH, adapter_name="oxen")
    pipeline.set_adapters(["oxen"], adapter_weights=[1.0])

    # 3. Keep model on GPU for fast inference (requires ~24GB+ VRAM)
    pipeline.to("cuda")
    
    # 4. Compile transformer for faster inference (20-40% speedup after warmup)
    pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
    
    # Clear any cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✅ Model loaded and ready for requests!")
    print("⚡ First inference will be slow (compiling), subsequent ones will be faster!")

@app.post("/edit")
async def edit_image(
    file: UploadFile = File(...), 
    prompt: str = Form(...),
    steps: int = Form(default=25),  # Can reduce to 15-20 for faster inference
):
    # 1. Read Image
    print(f"🖼️ Reading image from {file.filename}")
    image_data = await file.read()
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {prompt} (steps={steps})")
    
    # Clear memory before inference
    gc.collect()
    torch.cuda.empty_cache()
    
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=3.0,
            generator=torch.manual_seed(42)
        )
    
    result_image = output.images[0]

    # 3. Return Image
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

**What this does:**
- Model stays resident on GPU → no transfer overhead
- First request should drop from ~1:30 to ~10-20 seconds (mostly the 25 diffusion steps)

**Tradeoff:** Requires more VRAM (~20-30GB). If you hit OOM errors, you have options:

1. **Partial offload** - only offload the text encoder:
   ```python
   pipeline.to("cuda")
   pipeline.text_encoder.to("cpu")  # Smallest component, called once per request
   ```

2. **Sequential CPU offload** (slower but less VRAM than full GPU):
   ```python
   pipeline.enable_sequential_cpu_offload()
   ```

Restart the server and try again - you should see a big improvement!

```python
   pipeline.to("cuda")
   pipeline.text_encoder.to("cpu")  # Smallest component, called once per request
```

```python
   pipeline.enable_sequential_cpu_offload()
```

---

**User**

will the torch.compile need to happen again on first request?

---

**Cursor**

**Yes**, `torch.compile` will need to re-run on first request after restart. The compiled kernels are cached in memory and lost when the process dies.

**However**, PyTorch does cache compiled artifacts to disk (usually `~/.cache/torch/inductor/`). If everything matches (same model, input shapes, PyTorch/CUDA versions), it *may* reuse the cache and compile faster the second time.

But since you changed from CPU offload → full GPU, the execution graph is different, so it will likely do a full recompile anyway.

**To speed up development cycles**, you could temporarily disable compile while testing:

```python
# pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
```

Then re-enable it once you're happy with the setup. That way you're not waiting 5-15 min on every server restart.

```python
# pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
```

