# Implement water shader plan
_Exported on 12/12/2025 at 07:12:00 EST from Cursor (2.1.39)_

---

**User**

Implement the plan in @tasks/021_water_shader.md 

---

**Cursor**

I'll implement the water shader demo web app as specified in the task file. Let me start by checking the current state of the project and then scaffold the app.

I can see there's already an existing `app` directory with a similar stack. I'll create a new `water_shader_demo` directory for this water shader project. Let me scaffold the app:

Let me run with full permissions:

Now let me install dependencies and then customize the app:

Now let me create the custom components for the water shader demo. I'll create a modular structure with the WebGL shader infrastructure:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Water Shader Demo - Isometric NYC</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>

```

```css
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  /* Water-inspired color palette */
  --water-deep: #1a2a35;
  --water-mid: #4A6372;
  --water-light: #6b8a9a;
  --foam-color: #c8dce4;
  --bg-dark: #0d1419;
  --bg-card: #1a2630;
  --text-primary: #e8f0f4;
  --text-secondary: #8aa4b0;
  --accent: #5b9ab8;
  --grid-gap: 2px;
  --tile-size: 512px;
  
  font-family: 'JetBrains Mono', monospace;
  line-height: 1.5;
  font-weight: 400;

  color-scheme: dark;
  color: var(--text-primary);
  background-color: var(--bg-dark);

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: 
    radial-gradient(ellipse at 50% 0%, rgba(74, 99, 114, 0.15) 0%, transparent 60%),
    linear-gradient(180deg, var(--bg-dark) 0%, #0a1015 100%);
}

#root {
  width: 100%;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

```

```typescriptreact
import { useState } from 'react';
import { WaterShaderGrid } from './components/WaterShaderGrid';
import { ControlPanel } from './components/ControlPanel';
import './App.css';

export interface ShaderParams {
  waveSpeed: number;
  waveFrequency: number;
  foamThreshold: number;
  pixelSize: number;
}

const defaultParams: ShaderParams = {
  waveSpeed: 2.0,
  waveFrequency: 10.0,
  foamThreshold: 0.8,
  pixelSize: 256.0,
};

function App() {
  const [shaderParams, setShaderParams] = useState<ShaderParams>(defaultParams);
  const [showMask, setShowMask] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Water Shader Demo</h1>
        <p className="subtitle">Isometric NYC • 4×4 Quadrant Grid</p>
      </header>
      
      <main className="app-main">
        <WaterShaderGrid 
          rows={4} 
          cols={4} 
          tileSize={512}
          shaderParams={shaderParams}
          showMask={showMask}
        />
      </main>

      <aside className="app-sidebar">
        <ControlPanel 
          params={shaderParams}
          onParamsChange={setShaderParams}
          showMask={showMask}
          onShowMaskChange={setShowMask}
        />
      </aside>
    </div>
  );
}

export default App;

```

```css
.app {
  display: grid;
  grid-template-areas:
    "header header"
    "main sidebar";
  grid-template-columns: 1fr 320px;
  grid-template-rows: auto 1fr;
  min-height: 100vh;
  gap: 0;
}

.app-header {
  grid-area: header;
  padding: 24px 32px;
  border-bottom: 1px solid rgba(74, 99, 114, 0.3);
  background: linear-gradient(180deg, rgba(26, 38, 48, 0.8) 0%, transparent 100%);
}

.app-header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--text-primary);
}

.app-header .subtitle {
  margin: 4px 0 0 0;
  font-size: 0.75rem;
  color: var(--text-secondary);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.app-main {
  grid-area: main;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
  overflow: auto;
}

.app-sidebar {
  grid-area: sidebar;
  padding: 24px;
  background: var(--bg-card);
  border-left: 1px solid rgba(74, 99, 114, 0.3);
  overflow-y: auto;
}

@media (max-width: 1200px) {
  .app {
    grid-template-areas:
      "header"
      "main"
      "sidebar";
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr auto;
  }

  .app-sidebar {
    border-left: none;
    border-top: 1px solid rgba(74, 99, 114, 0.3);
  }
}

```

Now let me create the components directory and the main components:

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The blurred distance mask
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Threshold for foam appearance
  uniform float u_pixelSize;        // Pixelation factor for retro look
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Water color palette (from the design)
  const vec3 waterColor = vec3(0.29, 0.38, 0.44);    // #4A6372
  const vec3 waterDeep = vec3(0.10, 0.16, 0.21);     // Deeper water
  const vec3 foamColor = vec3(0.78, 0.86, 0.89);     // Light foam
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (distance to land)
    float distToLand = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(distToLand), 1.0);
      return;
    }
    
    // Determine if this pixel is water (based on the original color being close to water color)
    // This is a simple heuristic - in practice you'd use the mask more directly
    vec3 diff = abs(imageColor.rgb - waterColor);
    float isWater = step(length(diff), 0.3);
    
    // If not water, just show the original image
    if (isWater < 0.5) {
      gl_FragColor = imageColor;
      return;
    }
    
    // Create the wave driver
    // Combine time and distance - the "- distToLand" makes waves move toward shore
    float waveDriver = sin((u_time * u_waveSpeed) - (distToLand * u_waveFrequency));
    
    // Add a secondary wave for more natural movement
    float waveDriver2 = sin((u_time * u_waveSpeed * 0.7) - (distToLand * u_waveFrequency * 1.3) + 1.5);
    float combinedWave = (waveDriver + waveDriver2 * 0.5) / 1.5;
    
    // Pixelate/Quantize for retro look
    // This snaps the smooth sine wave into hard pixel bands
    float pixelWave = floor(combinedWave * 3.0) / 3.0;
    
    // Determine foam based on proximity to land and wave phase
    float foamMix = step(u_foamThreshold, distToLand + (pixelWave * 0.15));
    
    // Add some variation in water depth based on distance from shore
    vec3 waterBase = mix(waterDeep, waterColor, distToLand * 0.5 + 0.5);
    
    // Add subtle wave-based color variation
    waterBase += vec3(0.02) * pixelWave;
    
    // Mix water and foam
    vec3 finalColor = mix(waterBase, foamColor, foamMix * 0.7);
    
    // Keep original alpha
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext('webgl');
  if (!gl) {
    console.error('WebGL not supported');
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, 'a_position'),
      a_texCoord: gl.getAttribLocation(program, 'a_texCoord'),
      u_image: gl.getUniformLocation(program, 'u_image'),
      u_mask: gl.getUniformLocation(program, 'u_mask'),
      u_time: gl.getUniformLocation(program, 'u_time'),
      u_waveSpeed: gl.getUniformLocation(program, 'u_waveSpeed'),
      u_waveFrequency: gl.getUniformLocation(program, 'u_waveFrequency'),
      u_foamThreshold: gl.getUniformLocation(program, 'u_foamThreshold'),
      u_pixelSize: gl.getUniformLocation(program, 'u_pixelSize'),
      u_resolution: gl.getUniformLocation(program, 'u_resolution'),
      u_showMask: gl.getUniformLocation(program, 'u_showMask'),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);
      
      const i = (y * size + x) * 4;
      data[i] = value;     // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255;   // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescriptreact
import { useEffect, useRef, useCallback } from 'react';
import type { ShaderParams } from '../App';
import {
  initWebGL,
  createPlaceholderTexture,
  createGradientMaskTexture,
  createTexture,
} from '../shaders/water';
import './WaterTile.css';

interface WaterTileProps {
  row: number;
  col: number;
  size: number;
  imageSrc?: string;
  maskSrc?: string;
  shaderParams: ShaderParams;
  showMask: boolean;
}

export function WaterTile({
  row,
  col,
  size,
  imageSrc,
  maskSrc,
  shaderParams,
  showMask,
}: WaterTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<ReturnType<typeof initWebGL>>(null);
  const animationFrameRef = useRef<number>(0);
  const texturesRef = useRef<{
    image: WebGLTexture | null;
    mask: WebGLTexture | null;
  }>({ image: null, mask: null });
  const startTimeRef = useRef<number>(performance.now());

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = initWebGL(canvas);
    if (!context) return;

    glContextRef.current = context;
    const { gl, program, locations } = context;

    // Set up vertex buffer for a full-screen quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_position);
    gl.vertexAttribPointer(locations.a_position, 2, gl.FLOAT, false, 0, 0);

    // Set up texture coordinates
    const texCoords = new Float32Array([
      0, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 0,
    ]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_texCoord);
    gl.vertexAttribPointer(locations.a_texCoord, 2, gl.FLOAT, false, 0, 0);

    // Create placeholder textures
    texturesRef.current.image = createPlaceholderTexture(gl, [74, 99, 114, 255]);
    texturesRef.current.mask = createGradientMaskTexture(gl, size);

    gl.useProgram(program);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [size]);

  // Load image texture if provided
  useEffect(() => {
    if (!imageSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.image = createTexture(gl, img);
    };
    img.src = imageSrc;
  }, [imageSrc]);

  // Load mask texture if provided
  useEffect(() => {
    if (!maskSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.mask = createTexture(gl, img);
    };
    img.src = maskSrc;
  }, [maskSrc]);

  // Animation loop
  const render = useCallback(() => {
    const context = glContextRef.current;
    if (!context) return;

    const { gl, program, locations } = context;
    const { image, mask } = texturesRef.current;

    if (!image || !mask) return;

    gl.useProgram(program);

    // Update uniforms
    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    gl.uniform1f(locations.u_time, elapsed);
    gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
    gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
    gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
    gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
    gl.uniform2f(locations.u_resolution, size, size);
    gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, image);
    gl.uniform1i(locations.u_image, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, mask);
    gl.uniform1i(locations.u_mask, 1);

    // Draw
    gl.viewport(0, 0, size, size);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    animationFrameRef.current = requestAnimationFrame(render);
  }, [shaderParams, showMask, size]);

  // Start/restart animation when params change
  useEffect(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div className="water-tile" data-row={row} data-col={col}>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="water-tile-canvas"
      />
      <div className="water-tile-label">
        {row},{col}
      </div>
    </div>
  );
}

```

```css
.water-tile {
  position: relative;
  background: var(--bg-card);
  border: 1px solid rgba(74, 99, 114, 0.4);
  overflow: hidden;
}

.water-tile-canvas {
  display: block;
  width: 100%;
  height: 100%;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

.water-tile-label {
  position: absolute;
  top: 8px;
  left: 8px;
  background: rgba(13, 20, 25, 0.85);
  color: var(--text-secondary);
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.05em;
  pointer-events: none;
}

```

```typescriptreact
import { WaterTile } from './WaterTile';
import type { ShaderParams } from '../App';
import './WaterShaderGrid.css';

interface WaterShaderGridProps {
  rows: number;
  cols: number;
  tileSize: number;
  shaderParams: ShaderParams;
  showMask: boolean;
}

export function WaterShaderGrid({
  rows,
  cols,
  tileSize,
  shaderParams,
  showMask,
}: WaterShaderGridProps) {
  // Generate grid of tiles
  const tiles: Array<{ row: number; col: number }> = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      tiles.push({ row, col });
    }
  }

  // Calculate display size (scale down for UI)
  const displaySize = Math.min(128, tileSize / 4);

  return (
    <div
      className="water-shader-grid"
      style={{
        gridTemplateColumns: `repeat(${cols}, ${displaySize}px)`,
        gridTemplateRows: `repeat(${rows}, ${displaySize}px)`,
        gap: 'var(--grid-gap)',
      }}
    >
      {tiles.map(({ row, col }) => (
        <WaterTile
          key={`${row}-${col}`}
          row={row}
          col={col}
          size={tileSize}
          shaderParams={shaderParams}
          showMask={showMask}
        />
      ))}
    </div>
  );
}

```

```css
.water-shader-grid {
  display: grid;
  background: rgba(74, 99, 114, 0.15);
  padding: var(--grid-gap);
  border-radius: 8px;
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.4),
    0 0 0 1px rgba(74, 99, 114, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.water-shader-grid .water-tile {
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.water-shader-grid .water-tile:hover {
  transform: scale(1.02);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  z-index: 10;
}

```

```typescriptreact
import type { ShaderParams } from '../App';
import './ControlPanel.css';

interface ControlPanelProps {
  params: ShaderParams;
  onParamsChange: (params: ShaderParams) => void;
  showMask: boolean;
  onShowMaskChange: (show: boolean) => void;
}

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: SliderControlProps) {
  return (
    <div className="slider-control">
      <div className="slider-header">
        <label>{label}</label>
        <span className="slider-value">{value.toFixed(1)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

export function ControlPanel({
  params,
  onParamsChange,
  showMask,
  onShowMaskChange,
}: ControlPanelProps) {
  const updateParam = <K extends keyof ShaderParams>(
    key: K,
    value: ShaderParams[K]
  ) => {
    onParamsChange({ ...params, [key]: value });
  };

  return (
    <div className="control-panel">
      <h2>Shader Controls</h2>

      <div className="control-section">
        <h3>Wave Animation</h3>
        <SliderControl
          label="Wave Speed"
          value={params.waveSpeed}
          min={0.1}
          max={10}
          step={0.1}
          onChange={(v) => updateParam('waveSpeed', v)}
        />
        <SliderControl
          label="Wave Frequency"
          value={params.waveFrequency}
          min={1}
          max={30}
          step={0.5}
          onChange={(v) => updateParam('waveFrequency', v)}
        />
      </div>

      <div className="control-section">
        <h3>Foam Effect</h3>
        <SliderControl
          label="Foam Threshold"
          value={params.foamThreshold}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(v) => updateParam('foamThreshold', v)}
        />
      </div>

      <div className="control-section">
        <h3>Pixelation</h3>
        <SliderControl
          label="Pixel Size"
          value={params.pixelSize}
          min={32}
          max={512}
          step={16}
          onChange={(v) => updateParam('pixelSize', v)}
        />
      </div>

      <div className="control-section">
        <h3>Debug</h3>
        <label className="checkbox-control">
          <input
            type="checkbox"
            checked={showMask}
            onChange={(e) => onShowMaskChange(e.target.checked)}
          />
          <span>Show Distance Mask</span>
        </label>
      </div>

      <div className="control-section info-section">
        <h3>Info</h3>
        <p>
          This demo shows a 4×4 grid of 512×512 tiles with a WebGL water shader.
          The shader creates animated "crashing wave" effects using a distance
          field mask to detect proximity to shorelines.
        </p>
        <p>
          Currently using placeholder gradient masks. Real tiles and masks will
          be exported from the isometric NYC map data.
        </p>
      </div>
    </div>
  );
}

```

```css
.control-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.control-panel h2 {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-primary);
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(74, 99, 114, 0.3);
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.control-section h3 {
  margin: 0;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

/* Slider styles */
.slider-control {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.slider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.slider-header label {
  font-size: 0.75rem;
  color: var(--text-primary);
}

.slider-value {
  font-size: 0.7rem;
  color: var(--accent);
  background: rgba(91, 154, 184, 0.15);
  padding: 2px 8px;
  border-radius: 4px;
  font-variant-numeric: tabular-nums;
}

.slider-control input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  background: rgba(74, 99, 114, 0.4);
  border-radius: 3px;
  outline: none;
  cursor: pointer;
}

.slider-control input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 14px;
  height: 14px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.slider-control input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.2);
  box-shadow: 0 0 12px rgba(91, 154, 184, 0.5);
}

.slider-control input[type="range"]::-moz-range-thumb {
  width: 14px;
  height: 14px;
  background: var(--accent);
  border: none;
  border-radius: 50%;
  cursor: pointer;
}

/* Checkbox styles */
.checkbox-control {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  font-size: 0.75rem;
  color: var(--text-primary);
}

.checkbox-control input[type="checkbox"] {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(74, 99, 114, 0.6);
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
  transition: all 0.15s ease;
  position: relative;
}

.checkbox-control input[type="checkbox"]:checked {
  background: var(--accent);
  border-color: var(--accent);
}

.checkbox-control input[type="checkbox"]:checked::after {
  content: "✓";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: var(--bg-dark);
  font-size: 10px;
  font-weight: bold;
}

/* Info section */
.info-section {
  background: rgba(74, 99, 114, 0.1);
  padding: 16px;
  border-radius: 8px;
  margin-top: 8px;
}

.info-section p {
  margin: 0;
  font-size: 0.7rem;
  line-height: 1.6;
  color: var(--text-secondary);
}

.info-section p + p {
  margin-top: 8px;
}

```

Now let me start the dev server to test the app:

The server is running. Let me test it in the browser:

The app is working. I can see the 4×4 grid with WebGL shaders rendering the water effect with circular gradient masks. Let me scroll up to see the full layout and test the controls:

Let me resize the browser to see the full desktop layout:

Let me refresh and scroll to see the grid:

Excellent! The app is working beautifully! I can see:
- **Header**: "Water Shader Demo" with the subtitle
- **4×4 Grid**: 16 tiles with WebGL water shaders running
- **Wave Animation**: The circular gradient shows the distance mask effect with darker center (deep water) and lighter edges (shore)
- **Tile Labels**: Each tile is labeled with its grid position

Let me test the "Show Distance Mask" toggle:

The checkbox is checked. Let me scroll up to see the grid with the mask view enabled:

Let me resize to a larger width to see the side-by-side layout:

The water shader demo is working perfectly! I can see the full 4×4 grid with WebGL shaders animating the water effect. Let me clean up and summarize.

The water shader demo web app has been successfully scaffolded in `src/water_shader_demo/`. Here's a summary of what was built:

## Project Structure

```
src/water_shader_demo/
├── index.html                # Entry point with title
├── package.json              # Bun/Vite/React dependencies
├── vite.config.ts            # Vite configuration
├── src/
│   ├── main.tsx              # React entry point
│   ├── App.tsx               # Main app component
│   ├── App.css               # App layout styles
│   ├── index.css             # Global styles (water-inspired theme)
│   ├── components/
│   │   ├── WaterShaderGrid.tsx/css  # 4×4 grid container
│   │   ├── WaterTile.tsx/css        # Individual tile with WebGL canvas
│   │   └── ControlPanel.tsx/css     # UI controls for shader params
│   └── shaders/
│       └── water.ts          # WebGL shader code & utilities
└── public/
    └── tiles/                # (Ready for tile assets)
```

## Key Features

1. **WebGL Water Shader**: Implements the "blurred mask distance field" approach from the task spec
   - Animated waves that move toward shore
   - Foam effect at high-distance (shore) areas
   - Pixelation/quantization for retro pixel art look

2. **4×4 Tile Grid**: Displays 16 tiles with 512×512 WebGL canvases

3. **Interactive Controls**:
   - Wave Speed & Frequency sliders
   - Foam Threshold slider
   - Pixel Size slider
   - Debug toggle to view the distance mask

4. **Placeholder Masks**: Uses procedural radial gradient masks (center = deep water, edges = shore) until real tile data is exported

## Running the App

```bash
cd src/water_shader_demo
bun dev
# Opens at http://localhost:5174/
```

The app is now ready to receive the actual image tiles and distance mask tiles that will be exported from the isometric NYC map data. The shader is designed to accept `imageSrc` and `maskSrc` props to load real tile assets.

```plaintext
src/water_shader_demo/
├── index.html                # Entry point with title
├── package.json              # Bun/Vite/React dependencies
├── vite.config.ts            # Vite configuration
├── src/
│   ├── main.tsx              # React entry point
│   ├── App.tsx               # Main app component
│   ├── App.css               # App layout styles
│   ├── index.css             # Global styles (water-inspired theme)
│   ├── components/
│   │   ├── WaterShaderGrid.tsx/css  # 4×4 grid container
│   │   ├── WaterTile.tsx/css        # Individual tile with WebGL canvas
│   │   └── ControlPanel.tsx/css     # UI controls for shader params
│   └── shaders/
│       └── water.ts          # WebGL shader code & utilities
└── public/
    └── tiles/                # (Ready for tile assets)
```

```shellscript
cd src/water_shader_demo
bun dev
# Opens at http://localhost:5174/
```

---

**User**

OK - now update the app to use the single 0_0.png tile and mask in the public/tiles and public/masks dir - we no longer need a grid

---

**Cursor**

Good, the tile and mask files are already in place. Let me update the app to use a single tile instead of a grid:

```typescriptreact
import { useState } from 'react';
import { WaterTile } from './components/WaterTile';
import { ControlPanel } from './components/ControlPanel';
import './App.css';

export interface ShaderParams {
  waveSpeed: number;
  waveFrequency: number;
  foamThreshold: number;
  pixelSize: number;
}

const defaultParams: ShaderParams = {
  waveSpeed: 2.0,
  waveFrequency: 10.0,
  foamThreshold: 0.8,
  pixelSize: 256.0,
};

function App() {
  const [shaderParams, setShaderParams] = useState<ShaderParams>(defaultParams);
  const [showMask, setShowMask] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Water Shader Demo</h1>
        <p className="subtitle">Isometric NYC • Single Tile View</p>
      </header>
      
      <main className="app-main">
        <div className="tile-container">
          <WaterTile 
            size={512}
            imageSrc="/tiles/0_0.png"
            maskSrc="/masks/0_0.png"
            shaderParams={shaderParams}
            showMask={showMask}
          />
        </div>
      </main>

      <aside className="app-sidebar">
        <ControlPanel 
          params={shaderParams}
          onParamsChange={setShaderParams}
          showMask={showMask}
          onShowMaskChange={setShowMask}
        />
      </aside>
    </div>
  );
}

export default App;

```

```css
.app {
  display: grid;
  grid-template-areas:
    "header header"
    "main sidebar";
  grid-template-columns: 1fr 320px;
  grid-template-rows: auto 1fr;
  min-height: 100vh;
  gap: 0;
}

.app-header {
  grid-area: header;
  padding: 24px 32px;
  border-bottom: 1px solid rgba(74, 99, 114, 0.3);
  background: linear-gradient(180deg, rgba(26, 38, 48, 0.8) 0%, transparent 100%);
}

.app-header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--text-primary);
}

.app-header .subtitle {
  margin: 4px 0 0 0;
  font-size: 0.75rem;
  color: var(--text-secondary);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.app-main {
  grid-area: main;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
  overflow: auto;
}

.tile-container {
  background: rgba(74, 99, 114, 0.15);
  padding: 4px;
  border-radius: 8px;
  box-shadow:
    0 4px 24px rgba(0, 0, 0, 0.4),
    0 0 0 1px rgba(74, 99, 114, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.app-sidebar {
  grid-area: sidebar;
  padding: 24px;
  background: var(--bg-card);
  border-left: 1px solid rgba(74, 99, 114, 0.3);
  overflow-y: auto;
}

@media (max-width: 900px) {
  .app {
    grid-template-areas:
      "header"
      "main"
      "sidebar";
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr auto;
  }

  .app-sidebar {
    border-left: none;
    border-top: 1px solid rgba(74, 99, 114, 0.3);
  }
}

```

```typescriptreact
import { useEffect, useRef, useCallback } from 'react';
import type { ShaderParams } from '../App';
import {
  initWebGL,
  createPlaceholderTexture,
  createGradientMaskTexture,
  createTexture,
} from '../shaders/water';
import './WaterTile.css';

interface WaterTileProps {
  size: number;
  imageSrc?: string;
  maskSrc?: string;
  shaderParams: ShaderParams;
  showMask: boolean;
}

export function WaterTile({
  size,
  imageSrc,
  maskSrc,
  shaderParams,
  showMask,
}: WaterTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<ReturnType<typeof initWebGL>>(null);
  const animationFrameRef = useRef<number>(0);
  const texturesRef = useRef<{
    image: WebGLTexture | null;
    mask: WebGLTexture | null;
  }>({ image: null, mask: null });
  const startTimeRef = useRef<number>(performance.now());

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = initWebGL(canvas);
    if (!context) return;

    glContextRef.current = context;
    const { gl, program, locations } = context;

    // Set up vertex buffer for a full-screen quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_position);
    gl.vertexAttribPointer(locations.a_position, 2, gl.FLOAT, false, 0, 0);

    // Set up texture coordinates
    const texCoords = new Float32Array([
      0, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 0,
    ]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_texCoord);
    gl.vertexAttribPointer(locations.a_texCoord, 2, gl.FLOAT, false, 0, 0);

    // Create placeholder textures initially
    texturesRef.current.image = createPlaceholderTexture(gl, [74, 99, 114, 255]);
    texturesRef.current.mask = createGradientMaskTexture(gl, size);

    gl.useProgram(program);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [size]);

  // Load image texture if provided
  useEffect(() => {
    if (!imageSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.image = createTexture(gl, img);
    };
    img.src = imageSrc;
  }, [imageSrc]);

  // Load mask texture if provided
  useEffect(() => {
    if (!maskSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.mask = createTexture(gl, img);
    };
    img.src = maskSrc;
  }, [maskSrc]);

  // Animation loop
  const render = useCallback(() => {
    const context = glContextRef.current;
    if (!context) return;

    const { gl, program, locations } = context;
    const { image, mask } = texturesRef.current;

    if (!image || !mask) return;

    gl.useProgram(program);

    // Update uniforms
    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    gl.uniform1f(locations.u_time, elapsed);
    gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
    gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
    gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
    gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
    gl.uniform2f(locations.u_resolution, size, size);
    gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, image);
    gl.uniform1i(locations.u_image, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, mask);
    gl.uniform1i(locations.u_mask, 1);

    // Draw
    gl.viewport(0, 0, size, size);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    animationFrameRef.current = requestAnimationFrame(render);
  }, [shaderParams, showMask, size]);

  // Start/restart animation when params change
  useEffect(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div className="water-tile">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="water-tile-canvas"
      />
    </div>
  );
}

```

```css
.water-tile {
  position: relative;
  background: var(--bg-card);
  border: 1px solid rgba(74, 99, 114, 0.4);
  overflow: hidden;
  border-radius: 4px;
}

.water-tile-canvas {
  display: block;
  width: 512px;
  height: 512px;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

```

```typescriptreact
import type { ShaderParams } from '../App';
import './ControlPanel.css';

interface ControlPanelProps {
  params: ShaderParams;
  onParamsChange: (params: ShaderParams) => void;
  showMask: boolean;
  onShowMaskChange: (show: boolean) => void;
}

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: SliderControlProps) {
  return (
    <div className="slider-control">
      <div className="slider-header">
        <label>{label}</label>
        <span className="slider-value">{value.toFixed(1)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

export function ControlPanel({
  params,
  onParamsChange,
  showMask,
  onShowMaskChange,
}: ControlPanelProps) {
  const updateParam = <K extends keyof ShaderParams>(
    key: K,
    value: ShaderParams[K]
  ) => {
    onParamsChange({ ...params, [key]: value });
  };

  return (
    <div className="control-panel">
      <h2>Shader Controls</h2>

      <div className="control-section">
        <h3>Wave Animation</h3>
        <SliderControl
          label="Wave Speed"
          value={params.waveSpeed}
          min={0.1}
          max={10}
          step={0.1}
          onChange={(v) => updateParam('waveSpeed', v)}
        />
        <SliderControl
          label="Wave Frequency"
          value={params.waveFrequency}
          min={1}
          max={30}
          step={0.5}
          onChange={(v) => updateParam('waveFrequency', v)}
        />
      </div>

      <div className="control-section">
        <h3>Foam Effect</h3>
        <SliderControl
          label="Foam Threshold"
          value={params.foamThreshold}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(v) => updateParam('foamThreshold', v)}
        />
      </div>

      <div className="control-section">
        <h3>Pixelation</h3>
        <SliderControl
          label="Pixel Size"
          value={params.pixelSize}
          min={32}
          max={512}
          step={16}
          onChange={(v) => updateParam('pixelSize', v)}
        />
      </div>

      <div className="control-section">
        <h3>Debug</h3>
        <label className="checkbox-control">
          <input
            type="checkbox"
            checked={showMask}
            onChange={(e) => onShowMaskChange(e.target.checked)}
          />
          <span>Show Distance Mask</span>
        </label>
      </div>

      <div className="control-section info-section">
        <h3>Info</h3>
        <p>
          Viewing tile <code>0_0.png</code> with its corresponding distance mask.
          The shader creates animated "crashing wave" effects using the mask to
          detect proximity to shorelines.
        </p>
      </div>
    </div>
  );
}

```

Now let me remove the unused grid component files and test the app:

The actual tile is loading now! I can see the isometric NYC pixel art with buildings, the baseball diamond, trees, and the water area. Let me toggle the mask view to verify it's loading correctly:

The mask is loading correctly! I can see the blurred distance field:
- **Black** = land (no distance)
- **White** = open water (maximum distance)
- **Gradient** = shoreline zone where waves will animate

Let me turn off the mask view and scroll up to see the full layout:

Let me resize the browser to see the full side-by-side layout:

The app is now updated to show a single tile. Here's what I can see:
- The isometric NYC tile with park, baseball diamond, buildings, and water
- The shader controls panel below
- Both the tile image (`0_0.png`) and the distance mask are loading correctly

The app has been successfully updated:

## Changes Made

1. **`App.tsx`** - Removed grid component, now renders single `WaterTile` with:
   - `imageSrc="/tiles/0_0.png"`
   - `maskSrc="/masks/0_0.png"`
   - Updated subtitle to "Single Tile View"

2. **`App.css`** - Simplified layout with `.tile-container` wrapper

3. **`WaterTile.tsx`** - Removed row/col props and label, simplified to single tile view

4. **`WaterTile.css`** - Fixed 512×512 display size, removed label styles

5. **`ControlPanel.tsx`** - Updated info text to reference `0_0.png`

6. **Deleted** - `WaterShaderGrid.tsx` and `WaterShaderGrid.css` (no longer needed)

The tile and mask are both loading correctly - when you toggle "Show Distance Mask", you can see the blurred distance field that will drive the wave animation toward the shoreline.

---

**User**

ok - this is great, but as you can see the shader is messing up all of the colors of the original image - what we need is for all non-water pixels to be the same (black on the mask), and add a subtle lighter blue wave effect to the edge of the blue water pixels (white on the mask)

---

**Cursor**

You're right - the shader is incorrectly modifying all colors. I need to update the shader to:
1. Pass through non-water pixels (black mask) unchanged
2. Only apply the wave effect to water pixels (white mask)
3. Add foam at the shoreline transition

Let me fix the shader:

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors
  const vec3 foamColor = vec3(0.65, 0.78, 0.85);     // Light blue-white foam
  const vec3 waveHighlight = vec3(0.45, 0.58, 0.68); // Subtle lighter blue for waves
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue);
    
    // Add foam at shoreline
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.6);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

Now let me test the updated shader:

Now the shader is working correctly:

1. **Land areas** (buildings, park, baseball diamond, trees, roads) are showing their **original colors** - completely unaffected by the shader
2. **Water** on the right side shows the original water color
3. **Shoreline** has a subtle **lighter blue foam effect** visible as a band along the water's edge

The key changes to the shader:
- Uses mask value directly: `maskValue < 0.05` = land (pass through unchanged)
- Foam appears at the shoreline transition zone (where mask goes from black to white)
- Wave highlights are subtle and only affect water pixels
- Original image colors are preserved for all non-water areas

The wave animation should be creating subtle pulsing foam effects along the shoreline. The effect is intentionally subtle to maintain the pixel art aesthetic!

---

**User**

This is wonderful, but can we make it a bit more subtle? the wave color is too bright

---

**Cursor**

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue);
    
    // Add foam at shoreline
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.6);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

The wave effect is now much more subtle. The changes I made:

- **Foam color**: Darkened from bright `(0.65, 0.78, 0.85)` to muted `(0.42, 0.52, 0.58)`
- **Wave highlight**: Reduced from `(0.45, 0.58, 0.68)` to very subtle `(0.35, 0.45, 0.52)`
- **Foam intensity**: Reduced from `0.6` to `0.35`
- **Wave highlight intensity**: Added additional `0.5` multiplier

The water now retains its original color with just a gentle, barely-noticeable shimmer at the shoreline - much more in keeping with the pixel art aesthetic!

---

**User**

Amazing! Now can we add some very subtle "ripples" to the white portions of the pure water?

---

**Cursor**

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleLight = vec3(0.38, 0.48, 0.55);   // Slightly lighter for ripple peaks
  const vec3 rippleDark = vec3(0.26, 0.36, 0.42);    // Slightly darker for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * 0.12; // Very subtle
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - blend toward light or dark based on ripple value
    if (rippleIntensity > 0.0) {
      finalColor = mix(finalColor, rippleLight, rippleIntensity);
    } else {
      finalColor = mix(finalColor, rippleDark, -rippleIntensity * 0.7);
    }
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

The ripples are now added to the deep water areas. Looking at the water on the right side, you can see very subtle variations creating a gentle shimmer effect.

The ripple system I added:

- **3 overlapping wave patterns** at different angles (two diagonals + horizontal) for natural-looking water surface
- **Quantized** for pixel art aesthetic (creates distinct bands rather than smooth gradients)
- **Fades near shore** - ripples only appear in deep water (mask > 0.4), getting stronger toward open ocean
- **Very subtle intensity** (12%) with both lighter and slightly darker color variations
- **Slower animation** than the shore waves (30% of wave speed)

The effect creates a gentle, animated shimmer in the open water while preserving the shoreline foam effect!

---

**User**

great but can uou make the ripples darken the blue instead of lightening it a bit?

---

**Cursor**

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * 0.12; // Very subtle
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - blend toward light or dark based on ripple value
    if (rippleIntensity > 0.0) {
      finalColor = mix(finalColor, rippleLight, rippleIntensity);
    } else {
      finalColor = mix(finalColor, rippleDark, -rippleIntensity * 0.7);
    }
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * 0.12; // Very subtle
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

Done! The ripples now only darken the water instead of lightening it. The changes:

- Removed the light ripple color entirely
- Now only uses a darker blue `(0.22, 0.32, 0.38)` for the ripple effect
- Creates subtle dark bands that animate across the deep water

The effect creates gentle dark undulations in the open water, giving it that subtle depth without brightening the surface.

---

**User**

amazing, the noise for the ripples is a bit too predictable and repeated, can you make it a bit more chaotic?

---

**Cursor**

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create chaotic ripple pattern using many overlapping waves
    float rippleSpeed = u_waveSpeed * 0.25;
    
    // Multiple wave layers at different scales, angles, and speeds
    // Using prime-ish numbers and irrational-ish ratios to avoid repetition
    float r1 = sin((uv.x * 47.0 + uv.y * 53.0) + u_time * rippleSpeed * 1.0);
    float r2 = sin((uv.x * 31.0 - uv.y * 37.0) + u_time * rippleSpeed * 1.3 + 1.7);
    float r3 = sin((uv.x * 67.0 + uv.y * 23.0) + u_time * rippleSpeed * 0.7 + 3.1);
    float r4 = sin((uv.x * 19.0 - uv.y * 71.0) + u_time * rippleSpeed * 1.1 + 5.3);
    float r5 = sin((uv.x * 83.0 + uv.y * 41.0) + u_time * rippleSpeed * 0.9 + 2.3);
    
    // Smaller detail ripples
    float r6 = sin((uv.x * 127.0 - uv.y * 113.0) + u_time * rippleSpeed * 1.7 + 4.1) * 0.5;
    float r7 = sin((uv.x * 97.0 + uv.y * 89.0) + u_time * rippleSpeed * 0.6 + 6.7) * 0.4;
    
    // Combine with varying weights
    float combinedRipple = (r1 + r2 * 0.8 + r3 * 0.7 + r4 * 0.6 + r5 * 0.5 + r6 + r7) / 4.5;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * 0.12; // Very subtle
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

---

**User**

Can we also add a slider for darkening the water?

---

**Cursor**

I'll add a water darkening slider. Let me update the necessary files:

```typescriptreact
import { useState } from "react";
import { WaterTile } from "./components/WaterTile";
import { ControlPanel } from "./components/ControlPanel";
import "./App.css";

export interface ShaderParams {
  waveSpeed: number;
  waveFrequency: number;
  foamThreshold: number;
  pixelSize: number;
  rippleDarkness: number;
}

const defaultParams: ShaderParams = {
  waveSpeed: 2.0,
  waveFrequency: 10.0,
  foamThreshold: 0.8,
  pixelSize: 256.0,
  rippleDarkness: 0.12,
};

function App() {
  const [shaderParams, setShaderParams] = useState<ShaderParams>(defaultParams);
  const [showMask, setShowMask] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Water Shader Demo</h1>
        <p className="subtitle">Isometric NYC • Single Tile View</p>
      </header>

      <main className="app-main">
        <div className="tile-container">
          <WaterTile
            size={512}
            imageSrc="/tiles/0_0.png"
            maskSrc="/masks/0_0.png"
            shaderParams={shaderParams}
            showMask={showMask}
          />
        </div>
      </main>

      <aside className="app-sidebar">
        <ControlPanel
          params={shaderParams}
          onParamsChange={setShaderParams}
          showMask={showMask}
          onShowMaskChange={setShowMask}
        />
      </aside>
    </div>
  );
}

export default App;

```

```typescriptreact
import type { ShaderParams } from "../App";
import "./ControlPanel.css";

interface ControlPanelProps {
  params: ShaderParams;
  onParamsChange: (params: ShaderParams) => void;
  showMask: boolean;
  onShowMaskChange: (show: boolean) => void;
}

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: SliderControlProps) {
  return (
    <div className="slider-control">
      <div className="slider-header">
        <label>{label}</label>
        <span className="slider-value">{value.toFixed(1)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

export function ControlPanel({
  params,
  onParamsChange,
  showMask,
  onShowMaskChange,
}: ControlPanelProps) {
  const updateParam = <K extends keyof ShaderParams>(
    key: K,
    value: ShaderParams[K]
  ) => {
    onParamsChange({ ...params, [key]: value });
  };

  return (
    <div className="control-panel">
      <h2>Shader Controls</h2>

      <div className="control-section">
        <h3>Wave Animation</h3>
        <SliderControl
          label="Wave Speed"
          value={params.waveSpeed}
          min={0.1}
          max={10}
          step={0.1}
          onChange={(v) => updateParam("waveSpeed", v)}
        />
        <SliderControl
          label="Wave Frequency"
          value={params.waveFrequency}
          min={1}
          max={30}
          step={0.5}
          onChange={(v) => updateParam("waveFrequency", v)}
        />
      </div>

      <div className="control-section">
        <h3>Foam Effect</h3>
        <SliderControl
          label="Foam Threshold"
          value={params.foamThreshold}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(v) => updateParam("foamThreshold", v)}
        />
      </div>

      <div className="control-section">
        <h3>Ripples</h3>
        <SliderControl
          label="Ripple Darkness"
          value={params.rippleDarkness}
          min={0}
          max={0.5}
          step={0.01}
          onChange={(v) => updateParam("rippleDarkness", v)}
        />
      </div>

      <div className="control-section">
        <h3>Pixelation</h3>
        <SliderControl
          label="Pixel Size"
          value={params.pixelSize}
          min={32}
          max={512}
          step={16}
          onChange={(v) => updateParam("pixelSize", v)}
        />
      </div>

      <div className="control-section">
        <h3>Debug</h3>
        <label className="checkbox-control">
          <input
            type="checkbox"
            checked={showMask}
            onChange={(e) => onShowMaskChange(e.target.checked)}
          />
          <span>Show Distance Mask</span>
        </label>
      </div>

      <div className="control-section info-section">
        <h3>Info</h3>
        <p>
          Viewing tile <code>0_0.png</code> with its corresponding distance
          mask. The shader creates animated "crashing wave" effects using the
          mask to detect proximity to shorelines.
        </p>
      </div>
    </div>
  );
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create chaotic ripple pattern using many overlapping waves
    float rippleSpeed = u_waveSpeed * 0.25;
    
    // Multiple wave layers at different scales, angles, and speeds
    // Using prime-ish numbers and irrational-ish ratios to avoid repetition
    float r1 = sin((uv.x * 47.0 + uv.y * 53.0) + u_time * rippleSpeed * 1.0);
    float r2 = sin((uv.x * 31.0 - uv.y * 37.0) + u_time * rippleSpeed * 1.3 + 1.7);
    float r3 = sin((uv.x * 67.0 + uv.y * 23.0) + u_time * rippleSpeed * 0.7 + 3.1);
    float r4 = sin((uv.x * 19.0 - uv.y * 71.0) + u_time * rippleSpeed * 1.1 + 5.3);
    float r5 = sin((uv.x * 83.0 + uv.y * 41.0) + u_time * rippleSpeed * 0.9 + 2.3);
    
    // Smaller detail ripples
    float r6 = sin((uv.x * 127.0 - uv.y * 113.0) + u_time * rippleSpeed * 1.7 + 4.1) * 0.5;
    float r7 = sin((uv.x * 97.0 + uv.y * 89.0) + u_time * rippleSpeed * 0.6 + 6.7) * 0.4;
    
    // Combine with varying weights
    float combinedRipple = (r1 + r2 * 0.8 + r3 * 0.7 + r4 * 0.6 + r5 * 0.5 + r6 + r7) / 4.5;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * 0.12; // Very subtle
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create chaotic ripple pattern using many overlapping waves
    float rippleSpeed = u_waveSpeed * 0.25;
    
    // Multiple wave layers at different scales, angles, and speeds
    // Using prime-ish numbers and irrational-ish ratios to avoid repetition
    float r1 = sin((uv.x * 47.0 + uv.y * 53.0) + u_time * rippleSpeed * 1.0);
    float r2 = sin((uv.x * 31.0 - uv.y * 37.0) + u_time * rippleSpeed * 1.3 + 1.7);
    float r3 = sin((uv.x * 67.0 + uv.y * 23.0) + u_time * rippleSpeed * 0.7 + 3.1);
    float r4 = sin((uv.x * 19.0 - uv.y * 71.0) + u_time * rippleSpeed * 1.1 + 5.3);
    float r5 = sin((uv.x * 83.0 + uv.y * 41.0) + u_time * rippleSpeed * 0.9 + 2.3);
    
    // Smaller detail ripples
    float r6 = sin((uv.x * 127.0 - uv.y * 113.0) + u_time * rippleSpeed * 1.7 + 4.1) * 0.5;
    float r7 = sin((uv.x * 97.0 + uv.y * 89.0) + u_time * rippleSpeed * 0.6 + 6.7) * 0.4;
    
    // Combine with varying weights
    float combinedRipple = (r1 + r2 * 0.8 + r3 * 0.7 + r4 * 0.6 + r5 * 0.5 + r6 + r7) / 4.5;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
    a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
    },
```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create chaotic ripple pattern using many overlapping waves
    float rippleSpeed = u_waveSpeed * 0.25;
    
    // Multiple wave layers at different scales, angles, and speeds
    // Using prime-ish numbers and irrational-ish ratios to avoid repetition
    float r1 = sin((uv.x * 47.0 + uv.y * 53.0) + u_time * rippleSpeed * 1.0);
    float r2 = sin((uv.x * 31.0 - uv.y * 37.0) + u_time * rippleSpeed * 1.3 + 1.7);
    float r3 = sin((uv.x * 67.0 + uv.y * 23.0) + u_time * rippleSpeed * 0.7 + 3.1);
    float r4 = sin((uv.x * 19.0 - uv.y * 71.0) + u_time * rippleSpeed * 1.1 + 5.3);
    float r5 = sin((uv.x * 83.0 + uv.y * 41.0) + u_time * rippleSpeed * 0.9 + 2.3);
    
    // Smaller detail ripples
    float r6 = sin((uv.x * 127.0 - uv.y * 113.0) + u_time * rippleSpeed * 1.7 + 4.1) * 0.5;
    float r7 = sin((uv.x * 97.0 + uv.y * 89.0) + u_time * rippleSpeed * 0.6 + 6.7) * 0.4;
    
    // Combine with varying weights
    float combinedRipple = (r1 + r2 * 0.8 + r3 * 0.7 + r4 * 0.6 + r5 * 0.5 + r6 + r7) / 4.5;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescriptreact
import { useEffect, useRef, useCallback } from "react";
import type { ShaderParams } from "../App";
import {
  initWebGL,
  createPlaceholderTexture,
  createGradientMaskTexture,
  createTexture,
} from "../shaders/water";
import "./WaterTile.css";

interface WaterTileProps {
  size: number;
  imageSrc?: string;
  maskSrc?: string;
  shaderParams: ShaderParams;
  showMask: boolean;
}

export function WaterTile({
  size,
  imageSrc,
  maskSrc,
  shaderParams,
  showMask,
}: WaterTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<ReturnType<typeof initWebGL>>(null);
  const animationFrameRef = useRef<number>(0);
  const texturesRef = useRef<{
    image: WebGLTexture | null;
    mask: WebGLTexture | null;
  }>({ image: null, mask: null });
  const startTimeRef = useRef<number>(performance.now());

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = initWebGL(canvas);
    if (!context) return;

    glContextRef.current = context;
    const { gl, program, locations } = context;

    // Set up vertex buffer for a full-screen quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_position);
    gl.vertexAttribPointer(locations.a_position, 2, gl.FLOAT, false, 0, 0);

    // Set up texture coordinates
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_texCoord);
    gl.vertexAttribPointer(locations.a_texCoord, 2, gl.FLOAT, false, 0, 0);

    // Create placeholder textures initially
    texturesRef.current.image = createPlaceholderTexture(
      gl,
      [74, 99, 114, 255]
    );
    texturesRef.current.mask = createGradientMaskTexture(gl, size);

    gl.useProgram(program);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [size]);

  // Load image texture if provided
  useEffect(() => {
    if (!imageSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.image = createTexture(gl, img);
    };
    img.src = imageSrc;
  }, [imageSrc]);

  // Load mask texture if provided
  useEffect(() => {
    if (!maskSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.mask = createTexture(gl, img);
    };
    img.src = maskSrc;
  }, [maskSrc]);

  // Animation loop
  const render = useCallback(() => {
    const context = glContextRef.current;
    if (!context) return;

    const { gl, program, locations } = context;
    const { image, mask } = texturesRef.current;

    if (!image || !mask) return;

    gl.useProgram(program);

    // Update uniforms
    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    gl.uniform1f(locations.u_time, elapsed);
    gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
    gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
    gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
    gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
    gl.uniform2f(locations.u_resolution, size, size);
    gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);
    gl.uniform1f(locations.u_rippleDarkness, shaderParams.rippleDarkness);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, image);
    gl.uniform1i(locations.u_image, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, mask);
    gl.uniform1i(locations.u_mask, 1);

    // Draw
    gl.viewport(0, 0, size, size);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    animationFrameRef.current = requestAnimationFrame(render);
  }, [shaderParams, showMask, size]);

  // Start/restart animation when params change
  useEffect(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div className="water-tile">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="water-tile-canvas"
      />
    </div>
  );
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create chaotic ripple pattern using many overlapping waves
    float rippleSpeed = u_waveSpeed * 0.25;
    
    // Multiple wave layers at different scales, angles, and speeds
    // Using prime-ish numbers and irrational-ish ratios to avoid repetition
    float r1 = sin((uv.x * 47.0 + uv.y * 53.0) + u_time * rippleSpeed * 1.0);
    float r2 = sin((uv.x * 31.0 - uv.y * 37.0) + u_time * rippleSpeed * 1.3 + 1.7);
    float r3 = sin((uv.x * 67.0 + uv.y * 23.0) + u_time * rippleSpeed * 0.7 + 3.1);
    float r4 = sin((uv.x * 19.0 - uv.y * 71.0) + u_time * rippleSpeed * 1.1 + 5.3);
    float r5 = sin((uv.x * 83.0 + uv.y * 41.0) + u_time * rippleSpeed * 0.9 + 2.3);
    
    // Smaller detail ripples
    float r6 = sin((uv.x * 127.0 - uv.y * 113.0) + u_time * rippleSpeed * 1.7 + 4.1) * 0.5;
    float r7 = sin((uv.x * 97.0 + uv.y * 89.0) + u_time * rippleSpeed * 0.6 + 6.7) * 0.4;
    
    // Combine with varying weights
    float combinedRipple = (r1 + r2 * 0.8 + r3 * 0.7 + r4 * 0.6 + r5 * 0.5 + r6 + r7) / 4.5;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

Let me scroll down to see the new Ripples slider:

The new "Ripple Darkness" slider is now available under the **RIPPLES** section. You can adjust it from 0 (no ripples) to 0.5 (very dark ripples), with the default set at 0.12.

This lets you control exactly how dark the water ripples appear - slide it up for more pronounced ripples, or down for a subtler effect!

---

**User**

Can you go back to the original ripple shader? the new more chaotic one is a bit too choppy

---

**Cursor**

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

Done! Reverted to the original simpler ripple pattern with 3 overlapping waves (diagonal, reverse diagonal, and horizontal). This should give you the smoother, less choppy look while still providing nice water surface variation.

---

**User**

Can we add another slider for the *water* darkness? Not just the ripples, but I'd like to adjust the color of the water pixels

---

**Cursor**

```typescriptreact
import { useState } from "react";
import { WaterTile } from "./components/WaterTile";
import { ControlPanel } from "./components/ControlPanel";
import "./App.css";

export interface ShaderParams {
  waveSpeed: number;
  waveFrequency: number;
  foamThreshold: number;
  pixelSize: number;
  rippleDarkness: number;
  waterDarkness: number;
}

const defaultParams: ShaderParams = {
  waveSpeed: 2.0,
  waveFrequency: 10.0,
  foamThreshold: 0.8,
  pixelSize: 256.0,
  rippleDarkness: 0.12,
  waterDarkness: 0.0,
};

function App() {
  const [shaderParams, setShaderParams] = useState<ShaderParams>(defaultParams);
  const [showMask, setShowMask] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Water Shader Demo</h1>
        <p className="subtitle">Isometric NYC • Single Tile View</p>
      </header>

      <main className="app-main">
        <div className="tile-container">
          <WaterTile
            size={512}
            imageSrc="/tiles/0_0.png"
            maskSrc="/masks/0_0.png"
            shaderParams={shaderParams}
            showMask={showMask}
          />
        </div>
      </main>

      <aside className="app-sidebar">
        <ControlPanel
          params={shaderParams}
          onParamsChange={setShaderParams}
          showMask={showMask}
          onShowMaskChange={setShowMask}
        />
      </aside>
    </div>
  );
}

export default App;

```

```typescriptreact
import type { ShaderParams } from "../App";
import "./ControlPanel.css";

interface ControlPanelProps {
  params: ShaderParams;
  onParamsChange: (params: ShaderParams) => void;
  showMask: boolean;
  onShowMaskChange: (show: boolean) => void;
}

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: SliderControlProps) {
  return (
    <div className="slider-control">
      <div className="slider-header">
        <label>{label}</label>
        <span className="slider-value">{value.toFixed(1)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

export function ControlPanel({
  params,
  onParamsChange,
  showMask,
  onShowMaskChange,
}: ControlPanelProps) {
  const updateParam = <K extends keyof ShaderParams>(
    key: K,
    value: ShaderParams[K]
  ) => {
    onParamsChange({ ...params, [key]: value });
  };

  return (
    <div className="control-panel">
      <h2>Shader Controls</h2>

      <div className="control-section">
        <h3>Wave Animation</h3>
        <SliderControl
          label="Wave Speed"
          value={params.waveSpeed}
          min={0.1}
          max={10}
          step={0.1}
          onChange={(v) => updateParam("waveSpeed", v)}
        />
        <SliderControl
          label="Wave Frequency"
          value={params.waveFrequency}
          min={1}
          max={30}
          step={0.5}
          onChange={(v) => updateParam("waveFrequency", v)}
        />
      </div>

      <div className="control-section">
        <h3>Foam Effect</h3>
        <SliderControl
          label="Foam Threshold"
          value={params.foamThreshold}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(v) => updateParam("foamThreshold", v)}
        />
      </div>

      <div className="control-section">
        <h3>Water Color</h3>
        <SliderControl
          label="Water Darkness"
          value={params.waterDarkness}
          min={-0.3}
          max={0.3}
          step={0.01}
          onChange={(v) => updateParam("waterDarkness", v)}
        />
        <SliderControl
          label="Ripple Darkness"
          value={params.rippleDarkness}
          min={0}
          max={0.5}
          step={0.01}
          onChange={(v) => updateParam("rippleDarkness", v)}
        />
      </div>

      <div className="control-section">
        <h3>Pixelation</h3>
        <SliderControl
          label="Pixel Size"
          value={params.pixelSize}
          min={32}
          max={512}
          step={16}
          onChange={(v) => updateParam("pixelSize", v)}
        />
      </div>

      <div className="control-section">
        <h3>Debug</h3>
        <label className="checkbox-control">
          <input
            type="checkbox"
            checked={showMask}
            onChange={(e) => onShowMaskChange(e.target.checked)}
          />
          <span>Show Distance Mask</span>
        </label>
      </div>

      <div className="control-section info-section">
        <h3>Info</h3>
        <p>
          Viewing tile <code>0_0.png</code> with its corresponding distance
          mask. The shader creates animated "crashing wave" effects using the
          mask to detect proximity to shorelines.
        </p>
      </div>
    </div>
  );
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  uniform float u_waterDarkness;    // Controls overall water color darkness
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  uniform float u_waterDarkness;    // Controls overall water color darkness
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Apply overall water darkness adjustment
    // Negative values lighten, positive values darken
    finalColor = finalColor * (1.0 - u_waterDarkness);
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  uniform float u_waterDarkness;    // Controls overall water color darkness
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Apply overall water darkness adjustment
    // Negative values lighten, positive values darken
    finalColor = finalColor * (1.0 - u_waterDarkness);
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
    u_waterDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescript
// Vertex shader - simple pass-through for 2D rendering
export const vertexShaderSource = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader - water effect with crashing waves
// Mask semantics: Black (0.0) = land, White (1.0) = water
export const fragmentShaderSource = `
  precision mediump float;
  
  varying vec2 v_texCoord;
  
  uniform sampler2D u_image;        // The tile image texture
  uniform sampler2D u_mask;         // The distance mask (black=land, white=water)
  uniform float u_time;             // Animation time
  uniform float u_waveSpeed;        // Speed of wave animation
  uniform float u_waveFrequency;    // Frequency of waves
  uniform float u_foamThreshold;    // Controls how far from shore foam appears
  uniform float u_pixelSize;        // Not used currently
  uniform vec2 u_resolution;        // Canvas resolution
  uniform bool u_showMask;          // Debug: show mask instead of effect
  uniform float u_rippleDarkness;   // Controls how dark the ripples are
  uniform float u_waterDarkness;    // Controls overall water color darkness
  
  // Foam/wave colors - subtle variations
  const vec3 foamColor = vec3(0.42, 0.52, 0.58);     // Subtle lighter blue foam
  const vec3 waveHighlight = vec3(0.35, 0.45, 0.52); // Very subtle wave highlight
  const vec3 rippleDark = vec3(0.22, 0.32, 0.38);    // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the original image
    vec4 imageColor = texture2D(u_image, uv);
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), 1.0);
      return;
    }
    
    // If mask is very dark (land), pass through original pixel unchanged
    if (maskValue < 0.05) {
      gl_FragColor = imageColor;
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore (inverted mask)
    // maskValue near 0 = shore, maskValue near 1 = deep water
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    // Using (1.0 - shoreDistance) so waves originate from deep water and move to shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Create subtle ripple pattern for open water areas
    // Use multiple overlapping waves at different angles for natural look
    float rippleScale = 60.0; // Controls ripple size
    float rippleSpeed = u_waveSpeed * 0.3; // Slower than shore waves
    
    // Ripple pattern 1 - diagonal
    float ripple1 = sin((uv.x + uv.y) * rippleScale + u_time * rippleSpeed);
    // Ripple pattern 2 - other diagonal  
    float ripple2 = sin((uv.x - uv.y) * rippleScale * 0.8 + u_time * rippleSpeed * 1.3 + 1.0);
    // Ripple pattern 3 - horizontal drift
    float ripple3 = sin(uv.x * rippleScale * 0.6 + u_time * rippleSpeed * 0.7 + 2.5);
    
    // Combine ripples
    float combinedRipple = (ripple1 + ripple2 * 0.6 + ripple3 * 0.4) / 2.0;
    
    // Quantize ripples for pixel art look
    float pixelRipple = floor(combinedRipple * 3.0 + 0.5) / 3.0;
    
    // Ripples are stronger in deep water (high mask value), fade near shore
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam appears at the shoreline (where mask transitions from black to white)
    // We want foam when we're close to the shore but still in water
    float shoreProximity = 1.0 - maskValue; // High near shore, low in deep water
    
    // Foam threshold controls how far from shore the foam extends
    // foamThreshold of 0.8 means foam appears when shoreProximity > 0.2 (close to shore)
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    
    // Foam pulses with waves - appears when wave is high
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water (subtle color variation)
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.15);
    
    // Start with original water color from the image
    vec3 finalColor = imageColor.rgb;
    
    // Apply overall water darkness adjustment
    // Negative values lighten, positive values darken
    finalColor = finalColor * (1.0 - u_waterDarkness);
    
    // Add ripples in deep water - darken based on ripple value
    float darkRipple = max(0.0, rippleIntensity) + max(0.0, -rippleIntensity) * 0.5;
    finalColor = mix(finalColor, rippleDark, darkRipple);
    
    // Add subtle wave highlight in deeper water areas
    finalColor = mix(finalColor, waveHighlight, waveHighlightIntensity * maskValue * 0.5);
    
    // Add foam at shoreline (very subtle)
    finalColor = mix(finalColor, foamColor, foamIntensity * 0.35);
    
    gl_FragColor = vec4(finalColor, imageColor.a);
  }
`;

// Create and compile a shader
export function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

// Create a shader program
export function createProgram(
  gl: WebGLRenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: {
    a_position: number;
    a_texCoord: number;
    u_image: WebGLUniformLocation | null;
    u_mask: WebGLUniformLocation | null;
    u_time: WebGLUniformLocation | null;
    u_waveSpeed: WebGLUniformLocation | null;
    u_waveFrequency: WebGLUniformLocation | null;
    u_foamThreshold: WebGLUniformLocation | null;
    u_pixelSize: WebGLUniformLocation | null;
    u_resolution: WebGLUniformLocation | null;
    u_showMask: WebGLUniformLocation | null;
    u_rippleDarkness: WebGLUniformLocation | null;
    u_waterDarkness: WebGLUniformLocation | null;
  };
} | null {
  const gl = canvas.getContext("webgl");
  if (!gl) {
    console.error("WebGL not supported");
    return null;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  if (!vertexShader || !fragmentShader) return null;

  const program = createProgram(gl, vertexShader, fragmentShader);
  if (!program) return null;

  return {
    gl,
    program,
    locations: {
      a_position: gl.getAttribLocation(program, "a_position"),
      a_texCoord: gl.getAttribLocation(program, "a_texCoord"),
      u_image: gl.getUniformLocation(program, "u_image"),
      u_mask: gl.getUniformLocation(program, "u_mask"),
      u_time: gl.getUniformLocation(program, "u_time"),
      u_waveSpeed: gl.getUniformLocation(program, "u_waveSpeed"),
      u_waveFrequency: gl.getUniformLocation(program, "u_waveFrequency"),
      u_foamThreshold: gl.getUniformLocation(program, "u_foamThreshold"),
      u_pixelSize: gl.getUniformLocation(program, "u_pixelSize"),
      u_resolution: gl.getUniformLocation(program, "u_resolution"),
      u_showMask: gl.getUniformLocation(program, "u_showMask"),
      u_rippleDarkness: gl.getUniformLocation(program, "u_rippleDarkness"),
      u_waterDarkness: gl.getUniformLocation(program, "u_waterDarkness"),
    },
  };
}

// Create a texture from an image
export function createTexture(
  gl: WebGLRenderingContext,
  image: HTMLImageElement
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // Set texture parameters for non-power-of-2 textures
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); // Pixel art look

  return texture;
}

// Create a placeholder texture with a solid color
export function createPlaceholderTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [74, 99, 114, 255]
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    1,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(color)
  );

  return texture;
}

// Create a gradient mask texture (for testing without actual mask images)
export function createGradientMaskTexture(
  gl: WebGLRenderingContext,
  size: number = 512
): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) return null;

  // Create a radial gradient mask (center is deep water, edges are shore)
  const data = new Uint8Array(size * size * 4);
  const center = size / 2;
  const maxDist = Math.sqrt(2) * center;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - center;
      const dy = y - center;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normalized = Math.min(1, dist / maxDist);
      const value = Math.floor(normalized * 255);

      const i = (y * size + x) * 4;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    size,
    size,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data
  );

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  return texture;
}

```

```typescriptreact
import { useEffect, useRef, useCallback } from "react";
import type { ShaderParams } from "../App";
import {
  initWebGL,
  createPlaceholderTexture,
  createGradientMaskTexture,
  createTexture,
} from "../shaders/water";
import "./WaterTile.css";

interface WaterTileProps {
  size: number;
  imageSrc?: string;
  maskSrc?: string;
  shaderParams: ShaderParams;
  showMask: boolean;
}

export function WaterTile({
  size,
  imageSrc,
  maskSrc,
  shaderParams,
  showMask,
}: WaterTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<ReturnType<typeof initWebGL>>(null);
  const animationFrameRef = useRef<number>(0);
  const texturesRef = useRef<{
    image: WebGLTexture | null;
    mask: WebGLTexture | null;
  }>({ image: null, mask: null });
  const startTimeRef = useRef<number>(performance.now());

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = initWebGL(canvas);
    if (!context) return;

    glContextRef.current = context;
    const { gl, program, locations } = context;

    // Set up vertex buffer for a full-screen quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_position);
    gl.vertexAttribPointer(locations.a_position, 2, gl.FLOAT, false, 0, 0);

    // Set up texture coordinates
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(locations.a_texCoord);
    gl.vertexAttribPointer(locations.a_texCoord, 2, gl.FLOAT, false, 0, 0);

    // Create placeholder textures initially
    texturesRef.current.image = createPlaceholderTexture(
      gl,
      [74, 99, 114, 255]
    );
    texturesRef.current.mask = createGradientMaskTexture(gl, size);

    gl.useProgram(program);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [size]);

  // Load image texture if provided
  useEffect(() => {
    if (!imageSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.image = createTexture(gl, img);
    };
    img.src = imageSrc;
  }, [imageSrc]);

  // Load mask texture if provided
  useEffect(() => {
    if (!maskSrc || !glContextRef.current) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const { gl } = glContextRef.current!;
      texturesRef.current.mask = createTexture(gl, img);
    };
    img.src = maskSrc;
  }, [maskSrc]);

  // Animation loop
  const render = useCallback(() => {
    const context = glContextRef.current;
    if (!context) return;

    const { gl, program, locations } = context;
    const { image, mask } = texturesRef.current;

    if (!image || !mask) return;

    gl.useProgram(program);

    // Update uniforms
    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    gl.uniform1f(locations.u_time, elapsed);
    gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
    gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
    gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
    gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
    gl.uniform2f(locations.u_resolution, size, size);
    gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);
    gl.uniform1f(locations.u_rippleDarkness, shaderParams.rippleDarkness);
    gl.uniform1f(locations.u_waterDarkness, shaderParams.waterDarkness);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, image);
    gl.uniform1i(locations.u_image, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, mask);
    gl.uniform1i(locations.u_mask, 1);

    // Draw
    gl.viewport(0, 0, size, size);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    animationFrameRef.current = requestAnimationFrame(render);
  }, [shaderParams, showMask, size]);

  // Start/restart animation when params change
  useEffect(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div className="water-tile">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="water-tile-canvas"
      />
    </div>
  );
}

```

Let me scroll to show the Water Color controls:

Done! I've added a new **"Water Darkness"** slider under the **WATER COLOR** section. The slider ranges from:
- **-0.3** (lighter/brighter water)
- **0** (original color - default)
- **0.3** (darker water)

Both the Water Darkness and Ripple Darkness controls are now grouped together in the "Water Color" section. You can now independently control:
1. The overall water color brightness
2. The ripple effect intensity

---

**User**

Update the README.md to add a section about the water shader

---

**Cursor**

```markdown
# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York
City using the latest and greatest AI tools available.

## Project Structure

```
isometric-nyc/
├── src/
│   ├── isometric_nyc/    # Python tile generation pipeline
│   ├── app/              # Web viewer (React + Deck.gl)
│   └── web/              # 3D tiles viewer (Three.js)
├── synthetic_data/       # Training datasets
├── tile_plans/           # Tile generation plans
└── tasks/                # Task documentation
```

---

## Web Viewer

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Deck.gl** — WebGL tile rendering with OrthographicView
- **Bun** — Package management

### Running the Web App

```bash
cd src/app

# Install dependencies
bun install

# Start development server
bun run dev
```

The app will open at **http://localhost:3000**

---

## Water Shader Demo

A WebGL shader demo for adding animated water effects to the isometric map tiles.

### Features

- **Shoreline foam**: Subtle animated foam effect at water edges using a distance mask
- **Water ripples**: Gentle darkening ripples across open water areas
- **Adjustable parameters**: Real-time sliders for tweaking all effects
- **Pixel art aesthetic**: Quantized effects that maintain the retro look

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **WebGL** — Custom GLSL shaders for water animation
- **Bun** — Package management

### Running the Water Shader Demo

```bash
cd src/water_shader_demo

# Install dependencies
bun install

# Start development server
bun dev
```

The app will open at **http://localhost:5173** (or next available port)

### How It Works

The shader uses a **distance mask** approach:
- **Black pixels** in the mask = land (pass through unchanged)
- **White pixels** = deep water (receives ripple effects)
- **Gradient transition** = shoreline (receives foam effects)

The mask is a blurred version of the land/water boundary, providing smooth distance information for wave animation.

### Shader Controls

| Control | Description |
|---------|-------------|
| Wave Speed | Animation speed of shoreline waves |
| Wave Frequency | Density of wave bands |
| Foam Threshold | How far from shore foam appears |
| Water Darkness | Overall brightness of water pixels |
| Ripple Darkness | Intensity of deep water ripples |

### Adding Tile Assets

Place tile images and corresponding masks in:

```
src/water_shader_demo/public/
├── tiles/
│   └── 0_0.png          # The tile image
└── masks/
    └── 0_0.png          # Distance mask (black=land, white=water)
```

---

### Adding Tiles

Place your tile images in the `src/app/public/tiles/` directory:

```
src/app/public/tiles/
├── 0/                    # Zoom level 0 = native resolution (max zoom in)
│   ├── 0_0.png          # Tile at x=0, y=0
│   ├── 0_1.png          # Tile at x=0, y=1
│   ├── 1_0.png          # Tile at x=1, y=0
│   └── ...
├── 1/                    # Zoom level 1 = 2× zoomed out
├── 2/                    # Zoom level 2 = 4× zoomed out
└── info.json
```

**Tile naming:** `{zoom_level}/{x}_{y}.png`

**Zoom convention:** | Level | Resolution | Grid Size (20×20 base) |
|-------|------------|------------------------| | 0 | Native (512×512 per tile)
| 20×20 tiles | | 1 | 2× zoomed out | 10×10 tiles | | 2 | 4× zoomed out | 5×5
tiles |

---

## Python Pipeline

### Setup

Create a `.env` file in the root directory with API credentials:

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

Install dependencies using `uv`:

```bash
uv sync
```

### Scripts

```bash
# Generate tiles for an address
uv run isometric-nyc "350 5th Ave, New York, NY"

# Run the e2e generation viewer
uv run python src/isometric_nyc/e2e_generation/app.py

# Export tiles to the web app
**uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19**
```

### Exporting Tiles for the Web App

The `export_tiles_for_app.py` script exports quadrants from the SQLite database
to the web app's tile directory:

```bash
# Export a 20x20 region, normalizing coordinates so (0,0) is top-left
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19

# Export a specific region (e.g., quadrants 10-29 → tiles 0-19)
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 10,10 --br 29,29

# Export render images instead of generations
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --render

# Custom output directory
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --output-dir ./my-tiles/0

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --dry-run
```

**Coordinate normalization:** The `--tl` coordinate becomes `0_0.png` in the
output. For example, exporting `--tl 10,15 --br 20,25` produces tiles `0_0.png`
through `10_10.png`.

---

## Development

| Task               | Command                       |
| ------------------ | ----------------------------- |
| Run Python tests   | `uv run pytest`               |
| Format Python code | `uv run ruff format .`        |
| Lint Python code   | `uv run ruff check .`         |
| Run web app        | `cd src/app && bun run dev`   |
| Build web app      | `cd src/app && bun run build` |

```

```markdown
# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York
City using the latest and greatest AI tools available.

## Project Structure

```
isometric-nyc/
├── src/
│   ├── isometric_nyc/       # Python tile generation pipeline
│   ├── app/                 # Web viewer (React + Deck.gl)
│   ├── water_shader_demo/   # Water shader demo (React + WebGL)
│   └── web/                 # 3D tiles viewer (Three.js)
├── synthetic_data/          # Training datasets
├── tile_plans/              # Tile generation plans
└── tasks/                   # Task documentation
```

---

## Web Viewer

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Deck.gl** — WebGL tile rendering with OrthographicView
- **Bun** — Package management

### Running the Web App

```bash
cd src/app

# Install dependencies
bun install

# Start development server
bun run dev
```

The app will open at **http://localhost:3000**

---

## Water Shader Demo

A WebGL shader demo for adding animated water effects to the isometric map tiles.

### Features

- **Shoreline foam**: Subtle animated foam effect at water edges using a distance mask
- **Water ripples**: Gentle darkening ripples across open water areas
- **Adjustable parameters**: Real-time sliders for tweaking all effects
- **Pixel art aesthetic**: Quantized effects that maintain the retro look

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **WebGL** — Custom GLSL shaders for water animation
- **Bun** — Package management

### Running the Water Shader Demo

```bash
cd src/water_shader_demo

# Install dependencies
bun install

# Start development server
bun dev
```

The app will open at **http://localhost:5173** (or next available port)

### How It Works

The shader uses a **distance mask** approach:
- **Black pixels** in the mask = land (pass through unchanged)
- **White pixels** = deep water (receives ripple effects)
- **Gradient transition** = shoreline (receives foam effects)

The mask is a blurred version of the land/water boundary, providing smooth distance information for wave animation.

### Shader Controls

| Control | Description |
|---------|-------------|
| Wave Speed | Animation speed of shoreline waves |
| Wave Frequency | Density of wave bands |
| Foam Threshold | How far from shore foam appears |
| Water Darkness | Overall brightness of water pixels |
| Ripple Darkness | Intensity of deep water ripples |

### Adding Tile Assets

Place tile images and corresponding masks in:

```
src/water_shader_demo/public/
├── tiles/
│   └── 0_0.png          # The tile image
└── masks/
    └── 0_0.png          # Distance mask (black=land, white=water)
```

---

### Adding Tiles

Place your tile images in the `src/app/public/tiles/` directory:

```
src/app/public/tiles/
├── 0/                    # Zoom level 0 = native resolution (max zoom in)
│   ├── 0_0.png          # Tile at x=0, y=0
│   ├── 0_1.png          # Tile at x=0, y=1
│   ├── 1_0.png          # Tile at x=1, y=0
│   └── ...
├── 1/                    # Zoom level 1 = 2× zoomed out
├── 2/                    # Zoom level 2 = 4× zoomed out
└── info.json
```

**Tile naming:** `{zoom_level}/{x}_{y}.png`

**Zoom convention:** | Level | Resolution | Grid Size (20×20 base) |
|-------|------------|------------------------| | 0 | Native (512×512 per tile)
| 20×20 tiles | | 1 | 2× zoomed out | 10×10 tiles | | 2 | 4× zoomed out | 5×5
tiles |

---

## Python Pipeline

### Setup

Create a `.env` file in the root directory with API credentials:

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

Install dependencies using `uv`:

```bash
uv sync
```

### Scripts

```bash
# Generate tiles for an address
uv run isometric-nyc "350 5th Ave, New York, NY"

# Run the e2e generation viewer
uv run python src/isometric_nyc/e2e_generation/app.py

# Export tiles to the web app
**uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19**
```

### Exporting Tiles for the Web App

The `export_tiles_for_app.py` script exports quadrants from the SQLite database
to the web app's tile directory:

```bash
# Export a 20x20 region, normalizing coordinates so (0,0) is top-left
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19

# Export a specific region (e.g., quadrants 10-29 → tiles 0-19)
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 10,10 --br 29,29

# Export render images instead of generations
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --render

# Custom output directory
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --output-dir ./my-tiles/0

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --dry-run
```

**Coordinate normalization:** The `--tl` coordinate becomes `0_0.png` in the
output. For example, exporting `--tl 10,15 --br 20,25` produces tiles `0_0.png`
through `10_10.png`.

---

## Development

| Task               | Command                       |
| ------------------ | ----------------------------- |
| Run Python tests   | `uv run pytest`               |
| Format Python code | `uv run ruff format .`        |
| Lint Python code   | `uv run ruff check .`         |
| Run web app        | `cd src/app && bun run dev`   |
| Build web app      | `cd src/app && bun run build` |

```

```markdown
# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York
City using the latest and greatest AI tools available.

## Project Structure

```
isometric-nyc/
├── src/
│   ├── isometric_nyc/       # Python tile generation pipeline
│   ├── app/                 # Web viewer (React + Deck.gl)
│   ├── water_shader_demo/   # Water shader demo (React + WebGL)
│   └── web/                 # 3D tiles viewer (Three.js)
├── synthetic_data/          # Training datasets
├── tile_plans/              # Tile generation plans
└── tasks/                   # Task documentation
```

---

## Web Viewer

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Deck.gl** — WebGL tile rendering with OrthographicView
- **Bun** — Package management

### Running the Web App

```bash
cd src/app

# Install dependencies
bun install

# Start development server
bun run dev
```

The app will open at **http://localhost:3000**

---

## Water Shader Demo

A WebGL shader demo for adding animated water effects to the isometric map tiles.

### Features

- **Shoreline foam**: Subtle animated foam effect at water edges using a distance mask
- **Water ripples**: Gentle darkening ripples across open water areas
- **Adjustable parameters**: Real-time sliders for tweaking all effects
- **Pixel art aesthetic**: Quantized effects that maintain the retro look

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **WebGL** — Custom GLSL shaders for water animation
- **Bun** — Package management

### Running the Water Shader Demo

```bash
cd src/water_shader_demo

# Install dependencies
bun install

# Start development server
bun dev
```

The app will open at **http://localhost:5173** (or next available port)

### How It Works

The shader uses a **distance mask** approach:
- **Black pixels** in the mask = land (pass through unchanged)
- **White pixels** = deep water (receives ripple effects)
- **Gradient transition** = shoreline (receives foam effects)

The mask is a blurred version of the land/water boundary, providing smooth distance information for wave animation.

### Shader Controls

| Control | Description |
|---------|-------------|
| Wave Speed | Animation speed of shoreline waves |
| Wave Frequency | Density of wave bands |
| Foam Threshold | How far from shore foam appears |
| Water Darkness | Overall brightness of water pixels |
| Ripple Darkness | Intensity of deep water ripples |

### Adding Tile Assets

Place tile images and corresponding masks in:

```
src/water_shader_demo/public/
├── tiles/
│   └── 0_0.png          # The tile image
└── masks/
    └── 0_0.png          # Distance mask (black=land, white=water)
```

---

### Adding Tiles

Place your tile images in the `src/app/public/tiles/` directory:

```
src/app/public/tiles/
├── 0/                    # Zoom level 0 = native resolution (max zoom in)
│   ├── 0_0.png          # Tile at x=0, y=0
│   ├── 0_1.png          # Tile at x=0, y=1
│   ├── 1_0.png          # Tile at x=1, y=0
│   └── ...
├── 1/                    # Zoom level 1 = 2× zoomed out
├── 2/                    # Zoom level 2 = 4× zoomed out
└── info.json
```

**Tile naming:** `{zoom_level}/{x}_{y}.png`

**Zoom convention:** | Level | Resolution | Grid Size (20×20 base) |
|-------|------------|------------------------| | 0 | Native (512×512 per tile)
| 20×20 tiles | | 1 | 2× zoomed out | 10×10 tiles | | 2 | 4× zoomed out | 5×5
tiles |

---

## Python Pipeline

### Setup

Create a `.env` file in the root directory with API credentials:

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

Install dependencies using `uv`:

```bash
uv sync
```

### Scripts

```bash
# Generate tiles for an address
uv run isometric-nyc "350 5th Ave, New York, NY"

# Run the e2e generation viewer
uv run python src/isometric_nyc/e2e_generation/app.py

# Export tiles to the web app
**uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19**
```

### Exporting Tiles for the Web App

The `export_tiles_for_app.py` script exports quadrants from the SQLite database
to the web app's tile directory:

```bash
# Export a 20x20 region, normalizing coordinates so (0,0) is top-left
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19

# Export a specific region (e.g., quadrants 10-29 → tiles 0-19)
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 10,10 --br 29,29

# Export render images instead of generations
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --render

# Custom output directory
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --output-dir ./my-tiles/0

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --dry-run
```

**Coordinate normalization:** The `--tl` coordinate becomes `0_0.png` in the
output. For example, exporting `--tl 10,15 --br 20,25` produces tiles `0_0.png`
through `10_10.png`.

---

## Development

| Task                   | Command                                    |
| ---------------------- | ------------------------------------------ |
| Run Python tests       | `uv run pytest`                            |
| Format Python code     | `uv run ruff format .`                     |
| Lint Python code       | `uv run ruff check .`                      |
| Run web app            | `cd src/app && bun run dev`                |
| Build web app          | `cd src/app && bun run build`              |
| Run water shader demo  | `cd src/water_shader_demo && bun dev`      |
| Build water shader     | `cd src/water_shader_demo && bun run build`|

```

Done! I've updated the README.md with:

1. **Project Structure** - Added `water_shader_demo/` to the directory tree

2. **New "Water Shader Demo" section** including:
   - Features overview (shoreline foam, ripples, adjustable parameters, pixel art aesthetic)
   - Tech stack (React, TypeScript, Vite, WebGL, Bun)
   - Running instructions
   - How it works (distance mask explanation)
   - Shader controls table
   - Asset directory structure

3. **Development table** - Added water shader demo commands

