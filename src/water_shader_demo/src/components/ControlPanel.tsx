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
