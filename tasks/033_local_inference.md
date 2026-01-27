# Local Inference

We now have the ability to call model inference from a "local" server (actually hosted on Lambda Labs but port forwarded to localhost) via the following http API:

Curl Call
```
curl -X POST "http://localhost:8888/edit" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@<input-file.png>" \
     -F "prompt=Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)." \
     -F "steps=15" \
     -o <output-file.png>
```

Can we add a new model config type to the generation app in src/isometric_nyc/generation/app.py & viewer.js for "local" inference that uses this API instead of the Oxen API? We still need to support the oxen inference, just add another model config type and ensure that we can run inference with it.
