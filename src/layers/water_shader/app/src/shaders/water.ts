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
  uniform float u_maskOpacity;      // Debug: overlay mask with this opacity (0=hidden, 1=full)
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
    
    // Debug: overlay mask visualization if opacity > 0
    // Mask is shown as magenta tint (so it's visible over both dark and light areas)
    if (u_maskOpacity > 0.0) {
      vec3 maskColor = vec3(maskValue, 0.0, maskValue); // Magenta gradient
      finalColor = mix(finalColor, maskColor, u_maskOpacity);
    }
    
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
    u_maskOpacity: WebGLUniformLocation | null;
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
      u_maskOpacity: gl.getUniformLocation(program, "u_maskOpacity"),
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
