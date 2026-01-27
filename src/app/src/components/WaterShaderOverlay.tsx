import { useEffect, useRef, useState } from "react";
import type { ViewState } from "../App";
import {
  initWebGL,
  createTexture,
  createSolidTexture,
  type ShaderParams,
  type ShaderLocations,
} from "../shaders/water";
import { tilesBaseUrl } from "../config";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  maxZoomLevel: number;
}

interface WaterShaderOverlayProps {
  enabled: boolean;
  viewState: ViewState;
  tileConfig: TileConfig;
  shaderParams: ShaderParams;
  showMask?: boolean;
}

// Cache for loaded textures (module-level to persist across re-renders)
const imageCache = new Map<string, WebGLTexture>();
const maskCache = new Map<string, WebGLTexture>();
const pendingLoads = new Set<string>();
const failedLoads = new Set<string>();

export function WaterShaderOverlay({
  enabled,
  viewState,
  tileConfig,
  shaderParams,
  showMask = false,
}: WaterShaderOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<{
    gl: WebGLRenderingContext;
    program: WebGLProgram;
    locations: ShaderLocations;
  } | null>(null);
  const animationFrameRef = useRef<number>(0);
  const startTimeRef = useRef<number>(performance.now());
  const landTextureRef = useRef<WebGLTexture | null>(null);
  const buffersRef = useRef<{
    position: WebGLBuffer | null;
    texCoord: WebGLBuffer | null;
  }>({ position: null, texCoord: null });

  // Force re-render when textures load
  const [textureLoadCount, setTextureLoadCount] = useState(0);

  // Initialize WebGL context
  useEffect(() => {
    if (!enabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const context = initWebGL(canvas);
    if (!context) {
      console.error("WaterShaderOverlay: Failed to initialize WebGL");
      return;
    }

    console.log("WaterShaderOverlay: WebGL initialized successfully");
    glContextRef.current = context;
    const { gl, program } = context;

    // Create position buffer for a quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    buffersRef.current.position = positionBuffer;

    // Create texcoord buffer
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    buffersRef.current.texCoord = texCoordBuffer;

    // Create land (black) texture for tiles without masks
    landTextureRef.current = createSolidTexture(gl, [0, 0, 0, 255]);

    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const handleResize = () => {
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled]);

  // Load texture helper
  const loadTexture = (url: string, isImage: boolean) => {
    const cache = isImage ? imageCache : maskCache;

    if (cache.has(url) || pendingLoads.has(url) || failedLoads.has(url)) {
      return;
    }

    pendingLoads.add(url);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      pendingLoads.delete(url);
      const context = glContextRef.current;
      if (context) {
        const texture = createTexture(context.gl, img);
        if (texture) {
          cache.set(url, texture);
          setTextureLoadCount((c) => c + 1);
        }
      }
    };
    img.onerror = () => {
      pendingLoads.delete(url);
      failedLoads.add(url);
    };
    img.src = url;
  };

  // Animation loop
  useEffect(() => {
    if (!enabled) return;

    const render = () => {
      const context = glContextRef.current;
      const canvas = canvasRef.current;

      if (!context || !canvas) {
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const { gl, program, locations } = context;
      const { gridWidth, gridHeight, tileSize } = tileConfig;
      const { target, zoom } = viewState;
      const elapsed = (performance.now() - startTimeRef.current) / 1000;

      // Update canvas size if needed
      if (
        canvas.width !== window.innerWidth ||
        canvas.height !== window.innerHeight
      ) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }

      const scale = Math.pow(2, zoom);
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const totalHeight = gridHeight * tileSize;

      // Water shader always uses level 0 (base tiles) since masks only exist at that level
      // At very zoomed out views, we skip water effects entirely for performance
      const skipWaterEffects = zoom < -3; // Skip when very zoomed out

      if (skipWaterEffects) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      // Convert viewState from Y-up (bottom=0) to Y-down (top=0) for OSD compatibility
      // Our viewState.target[1] is in Y-up coordinates
      const viewCenterX = target[0];
      const viewCenterY_down = totalHeight - target[1]; // Convert to Y-down

      // Calculate visible world bounds in Y-down coordinates
      const worldLeft = viewCenterX - viewportWidth / (2 * scale);
      const worldRight = viewCenterX + viewportWidth / (2 * scale);
      const worldTop = viewCenterY_down - viewportHeight / (2 * scale);
      const worldBottom = viewCenterY_down + viewportHeight / (2 * scale);

      // Calculate visible tile range (always using base level 0 tiles)
      const startX = Math.max(0, Math.floor(worldLeft / tileSize));
      const endX = Math.min(gridWidth - 1, Math.ceil(worldRight / tileSize));
      const startY = Math.max(0, Math.floor(worldTop / tileSize));
      const endY = Math.min(gridHeight - 1, Math.ceil(worldBottom / tileSize));

      // Clear the canvas
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      // Render each visible tile (always using level 0)
      for (let tileX = startX; tileX <= endX; tileX++) {
        for (let tileY = startY; tileY <= endY; tileY++) {
          // Always use level 0 tiles for both images and masks
          const imageUrl = `${tilesBaseUrl}/tiles/0/${tileX}_${tileY}.png`;
          const maskUrl = `/water_masks/0/${tileX}_${tileY}.png`;

          // Get textures from cache
          const imageTexture = imageCache.get(imageUrl);
          const maskTexture = maskCache.get(maskUrl) || landTextureRef.current;

          // Start loading if not cached
          if (!imageTexture) {
            loadTexture(imageUrl, true);
            continue;
          }

          if (!maskCache.has(maskUrl) && !failedLoads.has(maskUrl)) {
            loadTexture(maskUrl, false);
          }

          if (!imageTexture || !maskTexture) continue;

          // Calculate screen position for this tile
          const tileWorldX = tileX * tileSize;
          const tileWorldY = tileY * tileSize; // Y-down coordinates

          // Convert to screen coordinates
          // Screen X: (worldX - viewCenterX) * scale + viewportWidth / 2
          // Screen Y: (worldY - viewCenterY_down) * scale + viewportHeight / 2
          // Note: WebGL canvas Y=0 is at bottom, so we need to flip for gl.viewport
          const screenX =
            (tileWorldX - viewCenterX) * scale + viewportWidth / 2;
          const screenY_fromTop =
            (tileWorldY - viewCenterY_down) * scale + viewportHeight / 2;
          const screenSize = tileSize * scale;

          // Convert to WebGL coordinates (Y=0 at bottom)
          const screenY = viewportHeight - screenY_fromTop - screenSize;

          // Skip if completely off-screen
          if (
            screenX + screenSize < 0 ||
            screenX > viewportWidth ||
            screenY + screenSize < 0 ||
            screenY > viewportHeight
          ) {
            continue;
          }

          // Set viewport and scissor for this tile
          gl.viewport(screenX, screenY, screenSize, screenSize);
          gl.scissor(screenX, screenY, screenSize, screenSize);
          gl.enable(gl.SCISSOR_TEST);

          // Bind buffers
          gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.position);
          gl.enableVertexAttribArray(locations.a_position);
          gl.vertexAttribPointer(
            locations.a_position,
            2,
            gl.FLOAT,
            false,
            0,
            0
          );

          gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.texCoord);
          gl.enableVertexAttribArray(locations.a_texCoord);
          gl.vertexAttribPointer(
            locations.a_texCoord,
            2,
            gl.FLOAT,
            false,
            0,
            0
          );

          // Set uniforms
          gl.uniform1f(locations.u_time, elapsed);
          gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
          gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
          gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
          gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
          gl.uniform2f(locations.u_resolution, screenSize, screenSize);
          gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);
          gl.uniform1f(locations.u_rippleDarkness, shaderParams.rippleDarkness);
          gl.uniform1f(locations.u_waterDarkness, shaderParams.waterDarkness);

          // Bind textures
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, imageTexture);
          gl.uniform1i(locations.u_image, 0);

          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, maskTexture);
          gl.uniform1i(locations.u_mask, 1);

          // Draw
          gl.drawArrays(gl.TRIANGLES, 0, 6);
          gl.disable(gl.SCISSOR_TEST);
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [
    enabled,
    viewState,
    tileConfig,
    shaderParams,
    showMask,
    textureLoadCount,
  ]);

  if (!enabled) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}
