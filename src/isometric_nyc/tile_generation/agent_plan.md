# Agent Plan

Your task is to debug an image generation workflow. The goal is to generate an
isometric pixel art image of a small section of New York City, in the style of
classic computer city builder games such as SimCity 3000, using a "template"
image with the left half already generated.

Your task is to iterate on the prompt logic / code in a script that generates
the image, incorporating any feedback from a separate script that checks the
output images to make changes to the prompt

This is what your workflow looks like:

1. If a `generation.png` exists in the directory and we have more than the first
   generate_tile script, then skip directly to step 3.

2. Run the latest generation script. Each script version must be saved in
   incremementing files starting with `generate_tile_001.py`. The command to run
   the script is
   `uv run python src/isometric_nyc/tile_generation/generate_tile_<nnn>.py`. If
   the generation script fails because of an error in the code, fix it. If the
   script fails because of a Gemini API error, re-run it.

The generation script will save the generated image `generation.png` in the
`tile_dir`.

3. Run the checker script. The command to run the checker script is
   `uv run python src/isometric_nyc/tile_generation/check_generation.py`. Thsi
   script will log any issues to stdout. Use these issues to decide what to do
   next.

4. Write the results of the run and checker to the `agent_log.md` file, in the
   following format:

```md
Generation: `generate_tile_nnn.py` Checker: \`\`\` {checker script output}
\`\`\`
```

IMPORTANT: Append these lines to the agent_log, do not overwrite it!

5. If the generation is good, then prompt the user for manual feedback. You may
   celebrate IF AND ONLY IF THE USER SAYS IT'S GOOD! If not, incorporate the
   feedback from the user to improve the prompt.

6. If the generation has issues, then we'll need to try something else. Create a
   copy of the generation script with the version number incrememented (e.g.
   `generate_tile_001.py` -> `generate_tile_002.py`) and update the script to
   try to address the issues with the checker.

You can update any part of the `generate_tile` function, including changing
which reference images are used, how they're supplied to the modelm, and any
optional processing you might want to do on them. However you MAY NOT change the
model (`gemini-3-pro-image-preview`) or how the output is saved.

After making the edits/changes to the next generation script, add a summary of
those next steps to the `agent_log.md` file in the following format:

```md
Next steps: {a description of the changes made to the next generation script}
```

IMPORTANT: Append this content to the agent_log, do not overwrite it!

Then, return to step 1 and run the next generation script.

## Context

You may use any of the following images in the tile dir:

1. `template.png` (MUST USE) - this is the masked tile with the left half of the
   image corresponding to a previous generation.
2. `render_256.png` - this is a 3D render of the actual city using the Google
   Maps 3D Tiles API - any generation must match the contents of this render,
   but it must not simply use the raw pixels from the render or just downsample
   it. The generation must match the pixel art style of the template image. It's
   downsampled to 256x256 to prevent the model from just copying the image data
   over to the masked half.
3. `whitebox.png` - this is a 3D "depth map" render of the building geometry in
   simple grayscale. It can be used as a depth/position reference.

## Important

This pipeline MUST BE GENERIC - that is, do not fix specific issues with the
specific generations. We need it to be able to work for ANY
template/render/whitebox for any location in the city with any buildings /
features.

DO NOT add any new reference images!
