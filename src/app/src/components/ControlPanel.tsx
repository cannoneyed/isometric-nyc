import { useState } from "react";
import type { ShaderParams } from "../shaders/water";

interface WaterShaderSettings {
	enabled: boolean;
	showMask: boolean;
	params: ShaderParams;
}

interface ControlPanelProps {
	waterShader: WaterShaderSettings;
	onWaterShaderChange: (settings: WaterShaderSettings) => void;
}

export function ControlPanel({
	waterShader,
	onWaterShaderChange,
}: ControlPanelProps) {
	const [collapsed, setCollapsed] = useState(false);

	return (
		<div className={`panel control-panel ${collapsed ? "collapsed" : ""}`}>
			<div
				className="panel-header"
				onClick={() => setCollapsed(!collapsed)}
				style={{ cursor: "pointer" }}
			>
				<span className="panel-title">Controls</span>
				<span className="collapse-icon">{collapsed ? "▶" : "▼"}</span>
			</div>

			{!collapsed && (
				<div className="panel-content">
					{/* Water shader controls */}
					<div className="control-group">
						<div className="control-label">
							<span>Water Shader</span>
							<label className="toggle-switch">
								<input
									type="checkbox"
									checked={waterShader.enabled}
									onChange={(e) =>
										onWaterShaderChange({
											...waterShader,
											enabled: e.target.checked,
										})
									}
								/>
								<span className="toggle-slider"></span>
							</label>
						</div>
					</div>

					{waterShader.enabled && (
						<>
							<div className="control-group">
								<div className="control-label">
									<span>Show Mask</span>
									<label className="toggle-switch">
										<input
											type="checkbox"
											checked={waterShader.showMask}
											onChange={(e) =>
												onWaterShaderChange({
													...waterShader,
													showMask: e.target.checked,
												})
											}
										/>
										<span className="toggle-slider"></span>
									</label>
								</div>
							</div>

							<div className="control-group">
								<div className="control-label">
									<span>Wave Speed</span>
									<span className="control-value">
										{waterShader.params.waveSpeed.toFixed(1)}
									</span>
								</div>
								<input
									type="range"
									min="0.5"
									max="5.0"
									step="0.1"
									value={waterShader.params.waveSpeed}
									onChange={(e) =>
										onWaterShaderChange({
											...waterShader,
											params: {
												...waterShader.params,
												waveSpeed: parseFloat(e.target.value),
											},
										})
									}
								/>
							</div>

							<div className="control-group">
								<div className="control-label">
									<span>Wave Frequency</span>
									<span className="control-value">
										{waterShader.params.waveFrequency.toFixed(1)}
									</span>
								</div>
								<input
									type="range"
									min="2.0"
									max="20.0"
									step="0.5"
									value={waterShader.params.waveFrequency}
									onChange={(e) =>
										onWaterShaderChange({
											...waterShader,
											params: {
												...waterShader.params,
												waveFrequency: parseFloat(e.target.value),
											},
										})
									}
								/>
							</div>

							<div className="control-group">
								<div className="control-label">
									<span>Ripple Intensity</span>
									<span className="control-value">
										{Math.round(waterShader.params.rippleDarkness * 100)}%
									</span>
								</div>
								<input
									type="range"
									min="0.0"
									max="1.0"
									step="0.05"
									value={waterShader.params.rippleDarkness}
									onChange={(e) =>
										onWaterShaderChange({
											...waterShader,
											params: {
												...waterShader.params,
												rippleDarkness: parseFloat(e.target.value),
											},
										})
									}
								/>
							</div>

							<div className="control-group">
								<div className="control-label">
									<span>Water Darkness</span>
									<span className="control-value">
										{Math.round(waterShader.params.waterDarkness * 100)}%
									</span>
								</div>
								<input
									type="range"
									min="0.0"
									max="0.5"
									step="0.05"
									value={waterShader.params.waterDarkness}
									onChange={(e) =>
										onWaterShaderChange({
											...waterShader,
											params: {
												...waterShader.params,
												waterDarkness: parseFloat(e.target.value),
											},
										})
									}
								/>
							</div>

							<div className="control-group">
								<div className="control-label">
									<span>Foam Threshold</span>
									<span className="control-value">
										{Math.round(waterShader.params.foamThreshold * 100)}%
									</span>
								</div>
								<input
									type="range"
									min="0.3"
									max="0.95"
									step="0.05"
									value={waterShader.params.foamThreshold}
									onChange={(e) =>
										onWaterShaderChange({
											...waterShader,
											params: {
												...waterShader.params,
												foamThreshold: parseFloat(e.target.value),
											},
										})
									}
								/>
							</div>
						</>
					)}
				</div>
			)}
		</div>
	);
}
