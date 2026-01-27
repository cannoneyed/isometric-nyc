import type { ViewState } from "../App";

interface TileInfoProps {
	hoveredTile: { x: number; y: number } | null;
	viewState: ViewState;
	// Origin offset: Tile (0,0) corresponds to database (originX, originY)
	originX: number;
	originY: number;
}

export function TileInfo({
	hoveredTile,
	viewState,
	originX,
	originY,
}: TileInfoProps) {
	const isVisible = hoveredTile !== null;

	// Calculate database coordinates from tile coordinates
	// Tile (x, y) -> Database (x + originX, y + originY)
	const dbX = hoveredTile ? hoveredTile.x + originX : 0;
	const dbY = hoveredTile ? hoveredTile.y + originY : 0;

	return (
		<div className={`panel tile-info ${isVisible ? "visible" : ""}`}>
			<div className="panel-header">
				<span className="panel-title">Tile Info</span>
			</div>

			<div className="tile-coords">
				<div className="coord">
					<span className="coord-label">X</span>
					<span className="coord-value">{hoveredTile ? dbX : "—"}</span>
				</div>
				<div className="coord">
					<span className="coord-label">Y</span>
					<span className="coord-value">{hoveredTile ? dbY : "—"}</span>
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
					Tile: ({hoveredTile.x}, {hoveredTile.y})
					<br />
					View center: ({Math.round(viewState.target[0])},{" "}
					{Math.round(viewState.target[1])})
				</div>
			)}
		</div>
	);
}
