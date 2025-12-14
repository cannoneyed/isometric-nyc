interface ScanlineSettings {
  enabled: boolean;
  count: number;
  opacity: number;
}

interface ControlPanelProps {
  zoom: number;
  lightDirection: [number, number, number];
  onLightDirectionChange: (direction: [number, number, number]) => void;
  visibleTiles: number;
  scanlines: ScanlineSettings;
  onScanlinesChange: (settings: ScanlineSettings) => void;
}

export function ControlPanel({
  zoom,
  lightDirection,
  onLightDirectionChange,
  visibleTiles,
  scanlines,
  onScanlinesChange,
}: ControlPanelProps) {
  // Convert light direction to azimuth/elevation for UI
  const [lx, ly, lz] = lightDirection;
  const azimuth = Math.atan2(ly, lx) * (180 / Math.PI);
  const elevation =
    Math.atan2(lz, Math.sqrt(lx * lx + ly * ly)) * (180 / Math.PI);

  const handleAzimuthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const az = parseFloat(e.target.value) * (Math.PI / 180);
    const el = elevation * (Math.PI / 180);
    const cosEl = Math.cos(el);
    onLightDirectionChange([
      Math.cos(az) * cosEl,
      Math.sin(az) * cosEl,
      Math.sin(el),
    ]);
  };

  const handleElevationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const az = azimuth * (Math.PI / 180);
    const el = parseFloat(e.target.value) * (Math.PI / 180);
    const cosEl = Math.cos(el);
    onLightDirectionChange([
      Math.cos(az) * cosEl,
      Math.sin(az) * cosEl,
      Math.sin(el),
    ]);
  };

  // Calculate scale factor from zoom
  const scale = Math.pow(2, zoom);
  const scalePercent = Math.round(scale * 100);

  return (
    <div className="panel control-panel">
      <div className="panel-header">
        <span className="panel-title">Controls</span>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-label">Zoom</div>
          <div className="stat-value">{zoom.toFixed(1)}</div>
        </div>
        <div className="stat-item">
          <div className="stat-label">Scale</div>
          <div className="stat-value accent">{scalePercent}%</div>
        </div>
        <div className="stat-item">
          <div className="stat-label">Tiles</div>
          <div className="stat-value">{visibleTiles}</div>
        </div>
        <div className="stat-item">
          <div className="stat-label">Grid</div>
          <div className="stat-value">20×20</div>
        </div>
      </div>

      {/* Light controls (for future shader effects) */}
      <div className="control-group" style={{ marginTop: 16 }}>
        <div className="control-label">
          <span>Light Azimuth</span>
          <span className="control-value">{Math.round(azimuth)}°</span>
        </div>
        <input
          type="range"
          min="-180"
          max="180"
          step="5"
          value={azimuth}
          onChange={handleAzimuthChange}
        />
      </div>

      <div className="control-group">
        <div className="control-label">
          <span>Light Elevation</span>
          <span className="control-value">{Math.round(elevation)}°</span>
        </div>
        <input
          type="range"
          min="10"
          max="80"
          step="5"
          value={elevation}
          onChange={handleElevationChange}
        />
      </div>

      {/* Scanline controls */}
      <div className="control-group" style={{ marginTop: 16 }}>
        <div className="control-label">
          <span>Scanlines</span>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={scanlines.enabled}
              onChange={(e) =>
                onScanlinesChange({ ...scanlines, enabled: e.target.checked })
              }
            />
            <span className="toggle-slider"></span>
          </label>
        </div>
      </div>

      {scanlines.enabled && (
        <>
          <div className="control-group">
            <div className="control-label">
              <span>Line Density</span>
              <span className="control-value">{scanlines.count}</span>
            </div>
            <input
              type="range"
              min="100"
              max="600"
              step="50"
              value={scanlines.count}
              onChange={(e) =>
                onScanlinesChange({
                  ...scanlines,
                  count: parseInt(e.target.value),
                })
              }
            />
          </div>

          <div className="control-group">
            <div className="control-label">
              <span>Intensity</span>
              <span className="control-value">
                {Math.round(scanlines.opacity * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.6"
              step="0.05"
              value={scanlines.opacity}
              onChange={(e) =>
                onScanlinesChange({
                  ...scanlines,
                  opacity: parseFloat(e.target.value),
                })
              }
            />
          </div>
        </>
      )}

      {/* Keyboard shortcuts */}
      <div className="shortcuts">
        <div className="shortcut">
          <span className="key">Scroll</span>
          <span>Zoom in/out</span>
        </div>
        <div className="shortcut">
          <span className="key">Drag</span>
          <span>Pan around</span>
        </div>
        <div className="shortcut">
          <span className="key">Dbl-click</span>
          <span>Zoom to point</span>
        </div>
      </div>
    </div>
  );
}
