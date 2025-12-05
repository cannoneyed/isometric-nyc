# Oxen API Inference

We're now going to create a copy of the @src/isometric_nyc/generate_tile_v2.py
script in src/isometric_nyc that uses a custom fine-tuned oxen.ai model to
generate an `infill` image

First, use the same logic to generate a "masked" image if the neighboring tile
has contents, and save it as 'infill.png' instead of 'template.png'

Next, we're going to use the oxen API to generate an image:

To start, we'll need to upload the image to Google Cloud in order to give the
oxen API a URL. I'll need instructions for how to set up the cloud project to
host images at a specific URL, but let's generate the code stubs that will
upload the `infill` image to cloud and get that image's URL

Then, we pass that to the Oxen API via an HTTP Request as follows:

```
curl -X POST \
-H "Authorization: Bearer <YOUR_TOKEN>" \
-H "Content-Type: application/json" \
-d '{
  "model": "cannoneyed-modern-salmon-unicorn",
  "input_image": "https://hub.oxen.ai/api/repos/ox/Oxen-Character-Simple-Vector-Graphic/file/main/images/reference/bloxy_white_bg.png",
  "prompt": "Convert the right side of the image to <isometric nyc pixel art> in precisely the style of the left side.",
  "num_inference_steps": 28
}' https://hub.oxen.ai/api/images/edit
```

YOUR_TOKEN is in the environment variable OXEN_INFILL_V02_API_KEY which you can
load with dotenv.

This request should yield an image URL - once that image URL has been finished,
download the image to `generation.png` in the tile_dir.

---

## Implementation Complete ✅

The script has been created at `src/isometric_nyc/generate_tile_oxen.py`

### Usage

```bash
uv run python src/isometric_nyc/generate_tile_oxen.py <tile_dir> --bucket <bucket_name>

# Or using the CLI entry point:
uv run generate-tile-oxen <tile_dir> --bucket <bucket_name>
```

### Google Cloud Storage Setup Instructions

To upload images to GCS and make them publicly accessible, follow these steps:

#### 1. Create a Google Cloud Project (if you don't have one)

```bash
# Install gcloud CLI if needed: https://cloud.google.com/sdk/docs/install
gcloud projects create isometric-nyc --name="Isometric NYC"
gcloud config set project isometric-nyc
```

#### 2. Enable the Cloud Storage API

```bash
gcloud services enable storage.googleapis.com
```

#### 3. Create a Storage Bucket

```bash
# Create a bucket (bucket names must be globally unique)
gcloud storage buckets create gs://isometric-nyc-infills --location=us-central1

# Make the bucket publicly readable
gcloud storage buckets add-iam-policy-binding gs://isometric-nyc-infills \
  --member=allUsers \
  --role=roles/storage.objectViewer
```

#### 4. Set Up Authentication

**Option A: Using Application Default Credentials (recommended for local dev)**

```bash
gcloud auth application-default login
```

**Option B: Using a Service Account (recommended for production/CI)**

```bash
# Create a service account
gcloud iam service-accounts create isometric-nyc-uploader \
  --display-name="Isometric NYC Image Uploader"

# Grant storage permissions
gcloud projects add-iam-policy-binding isometric-nyc \
  --member="serviceAccount:isometric-nyc-uploader@isometric-nyc.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"

# Create and download a key file
gcloud iam service-accounts keys create ./gcs-key.json \
  --iam-account=isometric-nyc-uploader@isometric-nyc.iam.gserviceaccount.com

# Set the environment variable in your .env file:
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json
```

#### 5. Environment Variables

Add these to your `.env` file:

```
OXEN_INFILL_V02_API_KEY=your_oxen_api_key_here
# Only needed if using service account authentication:
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json
```

### How It Works

1. **Create Infill Image**: Composites neighboring tile generations with the
   current tile's render
2. **Upload to GCS**: Uploads the infill image to a public GCS bucket
3. **Call Oxen API**: Sends the image URL to the Oxen fine-tuned model
4. **Download Result**: Saves the generated image to `generation.png`
