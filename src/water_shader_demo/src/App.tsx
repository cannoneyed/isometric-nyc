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
