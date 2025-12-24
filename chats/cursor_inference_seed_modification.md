# Inference seed modification
_Exported on 12/22/2025 at 19:08:56 EST from Cursor (2.2.43)_

---

**User**

How can I change the seed for the inference?

---

**Cursor**

The seed is currently hardcoded to `42` in both endpoints. You can see it on:

- **Line 113** (`/edit` endpoint): `generator=torch.manual_seed(42)`
- **Line 154** (`/edit-b64` endpoint): `generator=torch.manual_seed(42)`

You have a few options to change it:

### Option 1: Change the hardcoded value
Simply replace `42` with any other integer (e.g., `123`, `9999`, etc.)

### Option 2: Make it a request parameter
Add `seed` to your request models so clients can control it per-request. For example:

```python
# In EditRequest class (line 14-17):
class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int = 42  # Add this

# Then in the inference call, use:
generator=torch.manual_seed(req.seed)
```

### Option 3: Use an environment variable
```python
SEED = int(os.environ.get("INFERENCE_SEED", 42))
# Then use: generator=torch.manual_seed(SEED)
```

### Option 4: Random seed each time
If you want different results each request, don't set a seed at all (or use `None`):
```python
generator=None  # Will use random seed
```

Would you like me to implement any of these options for you?

```python
# In EditRequest class (line 14-17):
class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int = 42  # Add this

# Then in the inference call, use:
generator=torch.manual_seed(req.seed)
```

```python
SEED = int(os.environ.get("INFERENCE_SEED", 42))
# Then use: generator=torch.manual_seed(SEED)
```

```python
generator=None  # Will use random seed
```

---

**User**

make it None by default, and also allow me to pass it explicitly as a route param

---

**Cursor**

```python
import time
import torch
import uvicorn
import gc
import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import Response
from pydantic import BaseModel
from io import BytesIO
from PIL import Image


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int | None = None

# CORRECT LIBRARY: Use Diffusers, not Transformers
from diffusers import QwenImageEditPipeline 
from diffusers import FlowMatchEulerDiscreteScheduler

app = FastAPI()


@app.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    
    # Read the body to measure upload time (before endpoint processes it)
    body = await request.body()
    upload_time = (time.perf_counter() - t0) * 1000
    body_kb = len(body) / 1024
    print(f"📤 {request.method} {request.url.path} - received {body_kb:.1f}KB in {upload_time:.1f}ms")
    
    # Reconstruct request with the body we already read
    from starlette.requests import Request
    from starlette.datastructures import State
    
    async def receive():
        return {"type": "http.request", "body": body}
    
    request = Request(request.scope, receive)
    
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"🌐 {request.method} {request.url.path} completed in {elapsed:.1f}ms total (upload={upload_time:.1f}ms, processing={elapsed-upload_time:.1f}ms)")
    return response


# Global variables
pipeline = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit" 

pwd = os.getcwd()
ADAPTER_PATH = f"{pwd}/models/cannoneyed-rural-rose-dingo"

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


@app.post("/edit-b64")
async def edit_image_b64(req: EditRequest):
    """Faster endpoint using base64 JSON instead of multipart form."""
    timings = {}
    t_start = time.perf_counter()
    
    # 1. Decode base64 image
    t0 = time.perf_counter()
    image_data = base64.b64decode(req.image_b64)
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    timings["decode_image"] = time.perf_counter() - t0
    print(f"📥 Image decoded: {input_image.size}, {len(req.image_b64)/1024:.1f}KB b64 ({timings['decode_image']*1000:.1f}ms)")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {req.prompt} (steps={req.steps})")
    
    t0 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    timings["gc_cleanup"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=req.prompt,
            image=input_image,
            num_inference_steps=req.steps,
            guidance_scale=3.0,
            generator=torch.manual_seed(42)
        )
    timings["inference"] = time.perf_counter() - t0
    
    result_image = output.images[0]

    # 3. Return base64 encoded result
    t0 = time.perf_counter()
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    timings["encode_response"] = time.perf_counter() - t0
    
    timings["total"] = time.perf_counter() - t_start
    
    print(f"⏱️  Timings: decode={timings['decode_image']*1000:.1f}ms | "
          f"gc={timings['gc_cleanup']*1000:.1f}ms | "
          f"inference={timings['inference']*1000:.1f}ms | "
          f"encode={timings['encode_response']*1000:.1f}ms | "
          f"TOTAL={timings['total']*1000:.1f}ms")
    
    return {"image_b64": result_b64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

```python
import time
import torch
import uvicorn
import gc
import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import Response
from pydantic import BaseModel
from io import BytesIO
from PIL import Image


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int | None = None

# CORRECT LIBRARY: Use Diffusers, not Transformers
from diffusers import QwenImageEditPipeline 
from diffusers import FlowMatchEulerDiscreteScheduler

app = FastAPI()


@app.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    
    # Read the body to measure upload time (before endpoint processes it)
    body = await request.body()
    upload_time = (time.perf_counter() - t0) * 1000
    body_kb = len(body) / 1024
    print(f"📤 {request.method} {request.url.path} - received {body_kb:.1f}KB in {upload_time:.1f}ms")
    
    # Reconstruct request with the body we already read
    from starlette.requests import Request
    from starlette.datastructures import State
    
    async def receive():
        return {"type": "http.request", "body": body}
    
    request = Request(request.scope, receive)
    
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"🌐 {request.method} {request.url.path} completed in {elapsed:.1f}ms total (upload={upload_time:.1f}ms, processing={elapsed-upload_time:.1f}ms)")
    return response


# Global variables
pipeline = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit" 

pwd = os.getcwd()
ADAPTER_PATH = f"{pwd}/models/cannoneyed-rural-rose-dingo"

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
    seed: int | None = Form(default=None),  # None = random seed each time
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


@app.post("/edit-b64")
async def edit_image_b64(req: EditRequest):
    """Faster endpoint using base64 JSON instead of multipart form."""
    timings = {}
    t_start = time.perf_counter()
    
    # 1. Decode base64 image
    t0 = time.perf_counter()
    image_data = base64.b64decode(req.image_b64)
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    timings["decode_image"] = time.perf_counter() - t0
    print(f"📥 Image decoded: {input_image.size}, {len(req.image_b64)/1024:.1f}KB b64 ({timings['decode_image']*1000:.1f}ms)")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {req.prompt} (steps={req.steps})")
    
    t0 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    timings["gc_cleanup"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=req.prompt,
            image=input_image,
            num_inference_steps=req.steps,
            guidance_scale=3.0,
            generator=torch.manual_seed(42)
        )
    timings["inference"] = time.perf_counter() - t0
    
    result_image = output.images[0]

    # 3. Return base64 encoded result
    t0 = time.perf_counter()
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    timings["encode_response"] = time.perf_counter() - t0
    
    timings["total"] = time.perf_counter() - t_start
    
    print(f"⏱️  Timings: decode={timings['decode_image']*1000:.1f}ms | "
          f"gc={timings['gc_cleanup']*1000:.1f}ms | "
          f"inference={timings['inference']*1000:.1f}ms | "
          f"encode={timings['encode_response']*1000:.1f}ms | "
          f"TOTAL={timings['total']*1000:.1f}ms")
    
    return {"image_b64": result_b64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

```python
import time
import torch
import uvicorn
import gc
import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import Response
from pydantic import BaseModel
from io import BytesIO
from PIL import Image


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int | None = None

# CORRECT LIBRARY: Use Diffusers, not Transformers
from diffusers import QwenImageEditPipeline 
from diffusers import FlowMatchEulerDiscreteScheduler

app = FastAPI()


@app.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    
    # Read the body to measure upload time (before endpoint processes it)
    body = await request.body()
    upload_time = (time.perf_counter() - t0) * 1000
    body_kb = len(body) / 1024
    print(f"📤 {request.method} {request.url.path} - received {body_kb:.1f}KB in {upload_time:.1f}ms")
    
    # Reconstruct request with the body we already read
    from starlette.requests import Request
    from starlette.datastructures import State
    
    async def receive():
        return {"type": "http.request", "body": body}
    
    request = Request(request.scope, receive)
    
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"🌐 {request.method} {request.url.path} completed in {elapsed:.1f}ms total (upload={upload_time:.1f}ms, processing={elapsed-upload_time:.1f}ms)")
    return response


# Global variables
pipeline = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit" 

pwd = os.getcwd()
ADAPTER_PATH = f"{pwd}/models/cannoneyed-rural-rose-dingo"

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
    seed: int | None = Form(default=None),  # None = random seed each time
):
    # 1. Read Image
    print(f"🖼️ Reading image from {file.filename}")
    image_data = await file.read()
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {prompt} (steps={steps}, seed={seed})")
    
    # Clear memory before inference
    gc.collect()
    torch.cuda.empty_cache()
    
    generator = torch.manual_seed(seed) if seed is not None else None
    
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=3.0,
            generator=generator
        )
    
    result_image = output.images[0]

    # 3. Return Image
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/edit-b64")
async def edit_image_b64(req: EditRequest):
    """Faster endpoint using base64 JSON instead of multipart form."""
    timings = {}
    t_start = time.perf_counter()
    
    # 1. Decode base64 image
    t0 = time.perf_counter()
    image_data = base64.b64decode(req.image_b64)
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    timings["decode_image"] = time.perf_counter() - t0
    print(f"📥 Image decoded: {input_image.size}, {len(req.image_b64)/1024:.1f}KB b64 ({timings['decode_image']*1000:.1f}ms)")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {req.prompt} (steps={req.steps})")
    
    t0 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    timings["gc_cleanup"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=req.prompt,
            image=input_image,
            num_inference_steps=req.steps,
            guidance_scale=3.0,
            generator=torch.manual_seed(42)
        )
    timings["inference"] = time.perf_counter() - t0
    
    result_image = output.images[0]

    # 3. Return base64 encoded result
    t0 = time.perf_counter()
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    timings["encode_response"] = time.perf_counter() - t0
    
    timings["total"] = time.perf_counter() - t_start
    
    print(f"⏱️  Timings: decode={timings['decode_image']*1000:.1f}ms | "
          f"gc={timings['gc_cleanup']*1000:.1f}ms | "
          f"inference={timings['inference']*1000:.1f}ms | "
          f"encode={timings['encode_response']*1000:.1f}ms | "
          f"TOTAL={timings['total']*1000:.1f}ms")
    
    return {"image_b64": result_b64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

```python
import time
import torch
import uvicorn
import gc
import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import Response
from pydantic import BaseModel
from io import BytesIO
from PIL import Image


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int | None = None

# CORRECT LIBRARY: Use Diffusers, not Transformers
from diffusers import QwenImageEditPipeline 
from diffusers import FlowMatchEulerDiscreteScheduler

app = FastAPI()


@app.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    
    # Read the body to measure upload time (before endpoint processes it)
    body = await request.body()
    upload_time = (time.perf_counter() - t0) * 1000
    body_kb = len(body) / 1024
    print(f"📤 {request.method} {request.url.path} - received {body_kb:.1f}KB in {upload_time:.1f}ms")
    
    # Reconstruct request with the body we already read
    from starlette.requests import Request
    from starlette.datastructures import State
    
    async def receive():
        return {"type": "http.request", "body": body}
    
    request = Request(request.scope, receive)
    
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"🌐 {request.method} {request.url.path} completed in {elapsed:.1f}ms total (upload={upload_time:.1f}ms, processing={elapsed-upload_time:.1f}ms)")
    return response


# Global variables
pipeline = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit" 

pwd = os.getcwd()
ADAPTER_PATH = f"{pwd}/models/cannoneyed-rural-rose-dingo"

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
    seed: int | None = Form(default=None),  # None = random seed each time
):
    # 1. Read Image
    print(f"🖼️ Reading image from {file.filename}")
    image_data = await file.read()
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {prompt} (steps={steps}, seed={seed})")
    
    # Clear memory before inference
    gc.collect()
    torch.cuda.empty_cache()
    
    generator = torch.manual_seed(seed) if seed is not None else None
    
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=3.0,
            generator=generator
        )
    
    result_image = output.images[0]

    # 3. Return Image
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/edit-b64")
async def edit_image_b64(req: EditRequest):
    """Faster endpoint using base64 JSON instead of multipart form."""
    timings = {}
    t_start = time.perf_counter()
    
    # 1. Decode base64 image
    t0 = time.perf_counter()
    image_data = base64.b64decode(req.image_b64)
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    timings["decode_image"] = time.perf_counter() - t0
    print(f"📥 Image decoded: {input_image.size}, {len(req.image_b64)/1024:.1f}KB b64 ({timings['decode_image']*1000:.1f}ms)")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {req.prompt} (steps={req.steps}, seed={req.seed})")
    
    t0 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    timings["gc_cleanup"] = time.perf_counter() - t0
    
    generator = torch.manual_seed(req.seed) if req.seed is not None else None
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=req.prompt,
            image=input_image,
            num_inference_steps=req.steps,
            guidance_scale=3.0,
            generator=generator
        )
    timings["inference"] = time.perf_counter() - t0
    
    result_image = output.images[0]

    # 3. Return base64 encoded result
    t0 = time.perf_counter()
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    timings["encode_response"] = time.perf_counter() - t0
    
    timings["total"] = time.perf_counter() - t_start
    
    print(f"⏱️  Timings: decode={timings['decode_image']*1000:.1f}ms | "
          f"gc={timings['gc_cleanup']*1000:.1f}ms | "
          f"inference={timings['inference']*1000:.1f}ms | "
          f"encode={timings['encode_response']*1000:.1f}ms | "
          f"TOTAL={timings['total']*1000:.1f}ms")
    
    return {"image_b64": result_b64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

Done! Both endpoints now have `seed` as an optional parameter that defaults to `None` (random each time). Here's how to use it:

**`/edit` endpoint (multipart form):**
```bash
curl -X POST http://localhost:8080/edit \
  -F "file=@image.png" \
  -F "prompt=your prompt" \
  -F "seed=42"  # Optional - omit for random
```

**`/edit-b64` endpoint (JSON):**
```json
{
  "image_b64": "...",
  "prompt": "your prompt",
  "steps": 25,
  "seed": 42  // Optional - omit or set null for random
}
```

```shellscript
curl -X POST http://localhost:8080/edit \
  -F "file=@image.png" \
  -F "prompt=your prompt" \
  -F "seed=42"  # Optional - omit for random
```

```json
{
  "image_b64": "...",
  "prompt": "your prompt",
  "steps": 25,
  "seed": 42  // Optional - omit or set null for random
}
```

---

**User**

Also - when I send a "prompt", it seems like the torch compile step is happening again - the first orund of inference takes forever!

---

**Cursor**

```python
pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
```

This is a known issue with `torch.compile` using `mode="reduce-overhead"` - it uses CUDA graphs which are very sensitive to input shapes. Different prompt lengths or image sizes can trigger recompilation.

Here are some options:

1. **Switch to `mode="default"`** - More tolerant of shape changes, still gives speedup but less aggressive
2. **Add a warmup step at startup** - Pre-compile with representative inputs
3. **Disable compilation entirely** - If the overhead isn't worth it for your use case

I'd recommend option 1 or 2. Would you like me to:

**A)** Change to `mode="default"` (quick fix, still fast)

**B)** Add a warmup step that runs a dummy inference at startup to pre-compile (best for consistent performance)

**C)** Remove `torch.compile` entirely (simplest, but loses the 20-40% speedup)

Which would you prefer?

---

**User**

let's change the mode="default"

---

**Cursor**

```python
import time
import torch
import uvicorn
import gc
import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import Response
from pydantic import BaseModel
from io import BytesIO
from PIL import Image


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    steps: int = 25
    seed: int | None = None

# CORRECT LIBRARY: Use Diffusers, not Transformers
from diffusers import QwenImageEditPipeline 
from diffusers import FlowMatchEulerDiscreteScheduler

app = FastAPI()


@app.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    
    # Read the body to measure upload time (before endpoint processes it)
    body = await request.body()
    upload_time = (time.perf_counter() - t0) * 1000
    body_kb = len(body) / 1024
    print(f"📤 {request.method} {request.url.path} - received {body_kb:.1f}KB in {upload_time:.1f}ms")
    
    # Reconstruct request with the body we already read
    from starlette.requests import Request
    from starlette.datastructures import State
    
    async def receive():
        return {"type": "http.request", "body": body}
    
    request = Request(request.scope, receive)
    
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"🌐 {request.method} {request.url.path} completed in {elapsed:.1f}ms total (upload={upload_time:.1f}ms, processing={elapsed-upload_time:.1f}ms)")
    return response


# Global variables
pipeline = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit" 

pwd = os.getcwd()
ADAPTER_PATH = f"{pwd}/models/cannoneyed-rural-rose-dingo"

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
    pipeline.transformer = torch.compile(pipeline.transformer, mode="default")
    
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
    seed: int | None = Form(default=None),  # None = random seed each time
):
    # 1. Read Image
    print(f"🖼️ Reading image from {file.filename}")
    image_data = await file.read()
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {prompt} (steps={steps}, seed={seed})")
    
    # Clear memory before inference
    gc.collect()
    torch.cuda.empty_cache()
    
    generator = torch.manual_seed(seed) if seed is not None else None
    
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=3.0,
            generator=generator
        )
    
    result_image = output.images[0]

    # 3. Return Image
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/edit-b64")
async def edit_image_b64(req: EditRequest):
    """Faster endpoint using base64 JSON instead of multipart form."""
    timings = {}
    t_start = time.perf_counter()
    
    # 1. Decode base64 image
    t0 = time.perf_counter()
    image_data = base64.b64decode(req.image_b64)
    input_image = Image.open(BytesIO(image_data)).convert("RGB")
    timings["decode_image"] = time.perf_counter() - t0
    print(f"📥 Image decoded: {input_image.size}, {len(req.image_b64)/1024:.1f}KB b64 ({timings['decode_image']*1000:.1f}ms)")
    
    # 2. Run Inference
    print(f"🎨 Editing with prompt: {req.prompt} (steps={req.steps}, seed={req.seed})")
    
    t0 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    timings["gc_cleanup"] = time.perf_counter() - t0
    
    generator = torch.manual_seed(req.seed) if req.seed is not None else None
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=req.prompt,
            image=input_image,
            num_inference_steps=req.steps,
            guidance_scale=3.0,
            generator=generator
        )
    timings["inference"] = time.perf_counter() - t0
    
    result_image = output.images[0]

    # 3. Return base64 encoded result
    t0 = time.perf_counter()
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    timings["encode_response"] = time.perf_counter() - t0
    
    timings["total"] = time.perf_counter() - t_start
    
    print(f"⏱️  Timings: decode={timings['decode_image']*1000:.1f}ms | "
          f"gc={timings['gc_cleanup']*1000:.1f}ms | "
          f"inference={timings['inference']*1000:.1f}ms | "
          f"encode={timings['encode_response']*1000:.1f}ms | "
          f"TOTAL={timings['total']*1000:.1f}ms")
    
    return {"image_b64": result_b64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

Done! Changed to `mode="default"`. This should be much more tolerant of varying prompt lengths and image sizes without triggering recompilation. You'll still get compilation speedup, just without the CUDA graph rigidity.

