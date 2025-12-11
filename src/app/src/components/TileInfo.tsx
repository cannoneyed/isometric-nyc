import type { ViewState } from "../App";

interface TileInfoProps {
  hoveredTile: { x: number; y: number } | null;
  viewState: ViewState;
}

export function TileInfo({ hoveredTile, viewState }: TileInfoProps) {
  const isVisible = hoveredTile !== null;

  // Calculate world position from tile coordinates
  const worldX = hoveredTile ? hoveredTile.x * 512 : 0;
  const worldY = hoveredTile ? hoveredTile.y * 512 : 0;

  return (
    <div className={`panel tile-info ${isVisible ? "visible" : ""}`}>
      <div className="panel-header">
        <span className="panel-title">Tile Info</span>
      </div>

      <div className="tile-coords">
        <div className="coord">
          <span className="coord-label">X</span>
          <span className="coord-value">{hoveredTile?.x ?? "—"}</span>
        </div>
        <div className="coord">
          <span className="coord-label">Y</span>
          <span className="coord-value">{hoveredTile?.y ?? "—"}</span>
        </div>
      </div>

      {hoveredTile && (
        <div
          style={{
            marginTop: 12,
            fontSize: 10,
            color: "var(--color-text-muted)",
          }}
        >
          World: ({worldX}, {worldY})
          <br />
          View center: ({Math.round(viewState.target[0])},{" "}
          {Math.round(viewState.target[1])})
        </div>
      )}
    </div>
  );
}
