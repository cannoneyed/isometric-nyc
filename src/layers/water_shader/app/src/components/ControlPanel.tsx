import { useState, useEffect } from "react";
import type { ShaderParams, TileCoords } from "../App";
import "./ControlPanel.css";

interface ControlPanelProps {
	params: ShaderParams;
	onParamsChange: (params: ShaderParams) => void;
	maskOpacity: number;
	onMaskOpacityChange: (opacity: number) => void;
	coords: TileCoords;
	onCoordsChange: (coords: TileCoords) => void;
	availableTiles: [number, number][];
	availableMasks: [number, number][];
	hasMask: boolean;
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
	const inputId = `slider-${label.toLowerCase().replace(/\s+/g, "-")}`;
	return (
		<div className="slider-control">
			<div className="slider-header">
				<label htmlFor={inputId}>{label}</label>
				<span className="slider-value">{value.toFixed(1)}</span>
			</div>
			<input
				id={inputId}
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

interface CoordinateInputProps {
	label: string;
	value: number;
	onChange: (value: number) => void;
}

function CoordinateInput({ label, value, onChange }: CoordinateInputProps) {
	const [inputValue, setInputValue] = useState(value.toString());
	const inputId = `coord-${label.toLowerCase()}`;

	// Sync local state with prop
	useEffect(() => {
		setInputValue(value.toString());
	}, [value]);

	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const val = e.target.value;
		// Allow empty, minus sign, or valid integer (including negative)
		if (val === "" || val === "-" || /^-?\d+$/.test(val)) {
			setInputValue(val);
		}
	};

	const handleBlur = () => {
		// Don't parse if just a minus sign or empty
		if (inputValue === "-" || inputValue === "") {
			setInputValue(value.toString());
			return;
		}
		const parsed = parseInt(inputValue, 10);
		if (!Number.isNaN(parsed)) {
			onChange(parsed);
		} else {
			setInputValue(value.toString());
		}
	};

	const handleKeyDown = (e: React.KeyboardEvent) => {
		if (e.key === "Enter") {
			handleBlur();
		}
	};

	return (
		<div className="coordinate-input">
			<label htmlFor={inputId}>{label}</label>
			<input
				id={inputId}
				type="text"
				inputMode="numeric"
				pattern="-?[0-9]*"
				value={inputValue}
				onChange={handleChange}
				onBlur={handleBlur}
				onKeyDown={handleKeyDown}
			/>
		</div>
	);
}

export function ControlPanel({
	params,
	onParamsChange,
	maskOpacity,
	onMaskOpacityChange,
	coords,
	onCoordsChange,
	availableTiles,
	availableMasks,
	hasMask,
}: ControlPanelProps) {
	const updateParam = <K extends keyof ShaderParams>(
		key: K,
		value: ShaderParams[K],
	) => {
		onParamsChange({ ...params, [key]: value });
	};

	const handleXChange = (x: number) => {
		onCoordsChange({ ...coords, x });
	};

	const handleYChange = (y: number) => {
		onCoordsChange({ ...coords, y });
	};

	// Quick navigation to available tiles
	const handleQuickNav = (x: number, y: number) => {
		onCoordsChange({ x, y });
	};

	// Find tiles that have both generation and mask
	const tilesWithMasks = availableTiles.filter(([x, y]) =>
		availableMasks.some(([mx, my]) => mx === x && my === y),
	);

	return (
		<div className="control-panel">
			<h2>Shader Controls</h2>

			<div className="control-section">
				<h3>Tile Coordinates</h3>
				<div className="coordinate-inputs">
					<CoordinateInput
						label="X"
						value={coords.x}
						onChange={handleXChange}
					/>
					<CoordinateInput
						label="Y"
						value={coords.y}
						onChange={handleYChange}
					/>
				</div>
				<div className="mask-status">
					{hasMask ? (
						<span className="status-ok">✓ Mask available</span>
					) : (
						<span className="status-warning">⚠ No mask for this tile</span>
					)}
				</div>
			</div>

			{tilesWithMasks.length > 0 && (
				<div className="control-section">
					<h3>Quick Navigation</h3>
					<p className="section-hint">Tiles with both generation and mask:</p>
					<div className="quick-nav-grid">
						{tilesWithMasks.slice(0, 12).map(([x, y]) => (
							<button
								type="button"
								key={`${x}_${y}`}
								className={`quick-nav-btn ${coords.x === x && coords.y === y ? "active" : ""}`}
								onClick={() => handleQuickNav(x, y)}
							>
								({x}, {y})
							</button>
						))}
						{tilesWithMasks.length > 12 && (
							<span className="more-tiles">
								+{tilesWithMasks.length - 12} more
							</span>
						)}
					</div>
				</div>
			)}

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
				<SliderControl
					label="Mask Overlay"
					value={maskOpacity}
					min={0}
					max={1}
					step={0.05}
					onChange={onMaskOpacityChange}
				/>
				<p className="section-hint">
					Overlays the distance mask in magenta to visualize water detection.
				</p>
			</div>

			<div className="control-section info-section">
				<h3>Info</h3>
				<p>
					Viewing tile at{" "}
					<code>
						({coords.x}, {coords.y})
					</code>
					. The shader creates animated "crashing wave" effects using the mask
					to detect proximity to shorelines.
				</p>
				<p className="stats">
					Available: {availableTiles.length} tiles, {availableMasks.length}{" "}
					masks
				</p>
			</div>
		</div>
	);
}
