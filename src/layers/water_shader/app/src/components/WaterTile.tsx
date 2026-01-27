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
	maskOpacity: number;
}

export function WaterTile({
	size,
	imageSrc,
	maskSrc,
	shaderParams,
	maskOpacity,
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
			[74, 99, 114, 255],
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
		gl.uniform1f(locations.u_maskOpacity, maskOpacity);
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
	}, [shaderParams, maskOpacity, size]);

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
