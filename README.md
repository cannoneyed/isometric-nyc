# Isometric NYC

# isometric-nyc

![isometric-nyc](https://cannoneyed.com/img/projects/thumbnail-isometric-nyc.jpg)

This is the codebase for the [isometric.nyc](https://isometric.nyc) project - you can learn more at [cannoneyed.com/projects/isometric-nyc](https://cannoneyed.com/projects/isometric-nyc)

## The Idea

Growing up, I played a lot of video games, and my favorites were world building games like SimCity 2000 and Rollercoaster Tycoon. As a core millennial rapidly approaching middle age, I’m a sucker for the nostalgic vibes of those late 90s / early 2000s games. As I stared out at the city, I couldn’t help but imagine what it would look like in the style of those childhood memories.

So here’s the idea: Make a giant isometric pixel-art map of New York City. And I’m going to use it as an excuse to push hard on the limits of the latest and greatest generative models and coding agents.

## The Codebase (warning!)

This codebase was built entirely via collaboration with coding agents such as [`gemini-cli`](https://geminicli.com/), [`Claude Code`](https://code.claude.com/docs/en/overview), and [`Cursor`](https://cursor.com/). As such, the code probably sucks. Honestly, I've looked at less than 1% of it, and I didn't write any of it by hand. YMMV, but because this was partly an exercise in pushing "vibe-engineering" to its limits I bought fully into the "hands-off" approach and the results speak for themselves.

## Getting Started

The easiest way to get started is to run the web app with the production data served from R2.

```bash
cd src/app

# Install dependencies
bun install

# Start development server (pointed at R2 Production NYC tiles data)
USE_R2_NYC=true bun run dev
```

If you open the app at [http://localhost:3000](http://localhost:3000) you'll be able to interact with the real, production data.

## Docs

* **[Setup](docs/setup.md)** For setting up and downloading sample/full city data
* **[App](docs/app.md)** For the tiled image viewer web app
* **[Bounds](docs/bounds.md)** For the city bounds editor
* **[Data](docs/data.md)** A description of the various datasets and tile data management
* **[Deployment](docs/deployment.md)** How the production app is deployed
* **[Generation](docs/generation.md)** How to generate new tiles
* **[Inference](inference/README.md)** How to set up custom fine-tuned model inference on Modal
* **[Water Shader](docs/water_shader.md)** The WIP water shader documentation

## Project Structure

```
isometric-nyc/
├── src/
│   ├── app/                 # Web viewer (React + OpenSeaDragon)
│   ├── isometric_nyc/       # Core libraries + workflows
│   ├── demos/               # Demos for WIP features (e.g. Water Shader)
│   └── web/                 # 3D map tiles data viewer / renderer (Three.js)
└── tasks/                   # Tasks for agents
```


## Development

| Task                  | Command                                     |
| --------------------- | ------------------------------------------- |
| Run Python tests      | `uv run pytest`                             |
| Format Python code    | `uv run ruff format .`                      |
| Lint Python code      | `uv run ruff check .`                       |
| Run web app           | `cd src/app && bun run dev`                 |
| Build web app         | `cd src/app && bun run build`               |

---
