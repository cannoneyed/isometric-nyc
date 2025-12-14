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

// Fragment shader - water effect overlay
// Mask semantics: Black (0.0) = land, White (1.0) = water
// This shader renders as an overlay - transparent where there's no water effect
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
  
  // Foam/wave colors
  const vec3 foamColor = vec3(0.85, 0.92, 0.95);     // Bright white-blue foam (much lighter!)
  const vec3 waveHighlight = vec3(0.7, 0.8, 0.88);   // Light blue for wave crests
  const vec3 rippleDark = vec3(0.1, 0.18, 0.25);     // Darker blue for ripple troughs
  
  void main() {
    vec2 uv = v_texCoord;
    
    // Sample the mask (0 = land, 1 = water)
    float maskValue = texture2D(u_mask, uv).r;
    
    // Debug mode: show mask
    if (u_showMask) {
      gl_FragColor = vec4(vec3(maskValue), maskValue > 0.01 ? 0.8 : 0.0);
      return;
    }
    
    // If mask is very dark (land), output transparent - let base tiles show through
    if (maskValue < 0.02) {
      gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
      return;
    }
    
    // We're in water territory - calculate wave effects
    
    // Create wave animation based on distance from shore
    float shoreDistance = maskValue;
    
    // Wave driver - creates waves that move toward shore
    float wavePhase = u_time * u_waveSpeed - shoreDistance * u_waveFrequency;
    float wave = sin(wavePhase);
    
    // Secondary wave for more natural movement
    float wave2 = sin(wavePhase * 0.7 + 2.0);
    float combinedWave = (wave + wave2 * 0.4) / 1.4;
    
    // Quantize for pixel art look (creates distinct bands)
    float pixelWave = floor(combinedWave * 4.0 + 0.5) / 4.0;
    
    // === RIPPLES IN DEEP WATER ===
    // Use multiple overlapping waves at incommensurate frequencies for organic feel
    float rippleSpeed = u_waveSpeed * 0.25;
    
    // Primary ripple layers - different scales and directions
    float r1 = sin((uv.x * 47.0 + uv.y * 31.0) + u_time * rippleSpeed * 1.0);
    float r2 = sin((uv.x * 29.0 - uv.y * 43.0) + u_time * rippleSpeed * 0.7 + 1.5);
    float r3 = sin((uv.x * 17.0 + uv.y * 53.0) + u_time * rippleSpeed * 1.3 + 3.1);
    float r4 = sin((uv.y * 37.0 - uv.x * 23.0) + u_time * rippleSpeed * 0.9 + 2.2);
    
    // Secondary finer detail ripples
    float r5 = sin((uv.x * 71.0 + uv.y * 67.0) + u_time * rippleSpeed * 1.1 + 0.7) * 0.5;
    float r6 = sin((uv.x * 83.0 - uv.y * 79.0) + u_time * rippleSpeed * 0.8 + 4.2) * 0.4;
    
    // Add some position-based variation to break up repetition
    float posNoise = sin(uv.x * 11.0) * sin(uv.y * 13.0) * 0.3;
    
    // Combine with varying weights
    float combinedRipple = (r1 + r2 * 0.8 + r3 * 0.6 + r4 * 0.7 + r5 + r6 + posNoise) / 3.5;
    
    // Quantize for pixel art look but with more levels for subtlety
    float pixelRipple = floor(combinedRipple * 5.0 + 0.5) / 5.0;
    
    // Ripples are stronger in deep water
    float deepWaterFactor = smoothstep(0.4, 0.8, maskValue);
    float rippleIntensity = pixelRipple * deepWaterFactor * u_rippleDarkness;
    
    // Foam at shoreline
    float shoreProximity = 1.0 - maskValue;
    float foamZone = smoothstep(1.0 - u_foamThreshold, 1.0 - u_foamThreshold + 0.15, shoreProximity);
    float foamIntensity = foamZone * max(0.0, pixelWave);
    
    // Wave highlight in deeper water - brighter on wave peaks
    float waveHighlightIntensity = (1.0 - foamZone) * max(0.0, pixelWave * 0.4);
    
    // Calculate the overlay effect - this is what we add/blend on top of the original tile
    // We need both darkening (ripples) and lightening (foam, wave crests)
    
    vec3 overlayColor = vec3(0.0);
    float overlayAlpha = 0.0;
    
    // Base water tint - apply water darkness as an overall color shift
    if (u_waterDarkness > 0.0) {
      vec3 waterTint = vec3(0.08, 0.12, 0.18); // Slight blue-dark tint
      overlayColor = waterTint;
      overlayAlpha = u_waterDarkness * maskValue * 0.5;
    }
    
    // Ripples create alternating light and dark bands
    // Positive rippleIntensity = trough (darken), negative = crest (lighten)
    float rippleCrest = max(0.0, -rippleIntensity); // Light bands
    float rippleTrough = max(0.0, rippleIntensity);  // Dark bands
    
    // Add darkening from ripple troughs
    if (rippleTrough > 0.0) {
      overlayColor = mix(overlayColor, rippleDark, rippleTrough * 0.7);
      overlayAlpha = max(overlayAlpha, rippleTrough * 0.6);
    }
    
    // Add lightening from ripple crests (subtle highlight)
    if (rippleCrest > 0.0) {
      float crestBlend = rippleCrest * 0.5;
      overlayColor = mix(overlayColor, waveHighlight, crestBlend);
      overlayAlpha = max(overlayAlpha, crestBlend * 0.4);
    }
    
    // Wave highlights in deeper water - brighten on wave peaks
    if (waveHighlightIntensity > 0.0) {
      float highlightBlend = waveHighlightIntensity * maskValue * 0.6;
      overlayColor = mix(overlayColor, waveHighlight, highlightBlend);
      overlayAlpha = max(overlayAlpha, highlightBlend * 0.5);
    }
    
    // Foam at shoreline - this should be the brightest effect!
    if (foamIntensity > 0.0) {
      float foamBlend = foamIntensity * 0.85; // Strong foam visibility
      overlayColor = mix(overlayColor, foamColor, foamBlend);
      overlayAlpha = max(overlayAlpha, foamIntensity * 0.9); // High alpha for bright foam
    }
    
    // Fade alpha based on how much we're in water
    overlayAlpha *= smoothstep(0.02, 0.12, maskValue);
    
    gl_FragColor = vec4(overlayColor, overlayAlpha);
  }
`;

export interface ShaderParams {
  waveSpeed: number;
  waveFrequency: number;
  foamThreshold: number;
  pixelSize: number;
  rippleDarkness: number;
  waterDarkness: number;
}

export const defaultShaderParams: ShaderParams = {
  waveSpeed: 2.0,
  waveFrequency: 10.0,
  foamThreshold: 0.7,
  pixelSize: 256.0,
  rippleDarkness: 0.5,
  waterDarkness: 0.15,
};

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

export interface ShaderLocations {
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
}

// Initialize WebGL context and shader program
export function initWebGL(canvas: HTMLCanvasElement): {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  locations: ShaderLocations;
} | null {
  const gl = canvas.getContext("webgl", {
    alpha: true,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
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
  image: HTMLImageElement | ImageBitmap
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

// Create a placeholder texture with a solid color (for land/missing masks)
export function createSolidTexture(
  gl: WebGLRenderingContext,
  color: [number, number, number, number] = [0, 0, 0, 255] // Black = land
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
