import { useState, useEffect, useCallback } from "react";
import "./App.css";

export interface TileCoords {
  x: number;
  y: number;
}

// Parse coordinates from URL query params
function getCoordsFromUrl(): TileCoords {
  const params = new URLSearchParams(window.location.search);
  const x = parseInt(params.get("x") || "0", 10);
  const y = parseInt(params.get("y") || "0", 10);
  return { x: Number.isNaN(x) ? 0 : x, y: Number.isNaN(y) ? 0 : y };
}

// Update URL with new coordinates
function updateUrlCoords(coords: TileCoords): void {
  const params = new URLSearchParams(window.location.search);
  params.set("x", coords.x.toString());
  params.set("y", coords.y.toString());
  const newUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, "", newUrl);
}

// API base - with Vite proxy configured, we can always use relative paths
function getApiBase(): string {
  return "";
}

function App() {
  const [coords, setCoords] = useState<TileCoords>(getCoordsFromUrl);
  const [inputX, setInputX] = useState<string>(coords.x.toString());
  const [inputY, setInputY] = useState<string>(coords.y.toString());
  const [inputUrl, setInputUrl] = useState<string | null>(null);
  const [generationUrl, setGenerationUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableTiles, setAvailableTiles] = useState<[number, number][]>([]);

  const apiBase = getApiBase();

  // Fetch available tiles on mount
  useEffect(() => {
    async function fetchAvailable() {
      try {
        const res = await fetch(`${apiBase}/api/available-tiles`);
        if (res.ok) {
          const data = await res.json();
          setAvailableTiles(data.tiles || []);
        }
      } catch (e) {
        console.warn("Could not fetch available tiles:", e);
      }
    }

    fetchAvailable();
  }, [apiBase]);

  // Load input and generation when coordinates change
  const loadImages = useCallback(
    async (newCoords: TileCoords) => {
      setLoading(true);
      setError(null);

      const inputApiUrl = `${apiBase}/api/input/${newCoords.x}/${newCoords.y}`;
      const generationApiUrl = `${apiBase}/api/generation/${newCoords.x}/${newCoords.y}`;

      try {
        // Check if input exists
        const inputRes = await fetch(inputApiUrl, { method: "HEAD" });
        if (!inputRes.ok) {
          const contentType = inputRes.headers.get("content-type");
          if (contentType?.includes("application/json")) {
            const err = await (await fetch(inputApiUrl)).json();
            throw new Error(
              err.error ||
                `Input not found at (${newCoords.x}, ${newCoords.y})`,
            );
          } else {
            throw new Error(
              "Backend server not running. Start it with: uv run python src/layers/snow/app/server.py --input_dir <input_dir> --generation_dir <gen_dir>",
            );
          }
        }
        setInputUrl(inputApiUrl);

        // Check if generation exists (optional)
        const generationRes = await fetch(generationApiUrl, { method: "HEAD" });
        if (generationRes.ok) {
          setGenerationUrl(generationApiUrl);
        } else {
          setGenerationUrl(null);
          console.warn(
            `No generation available for (${newCoords.x}, ${newCoords.y})`,
          );
        }

        setError(null);
      } catch (e) {
        if (e instanceof TypeError && e.message.includes("fetch")) {
          setError(
            "Cannot connect to backend. Make sure the server is running on port 5002.",
          );
        } else {
          setError(e instanceof Error ? e.message : "Failed to load images");
        }
        setInputUrl(null);
        setGenerationUrl(null);
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  // Load images when coords change
  useEffect(() => {
    loadImages(coords);
  }, [coords, loadImages]);

  // Handle coordinate change
  const handleCoordsChange = useCallback((newCoords: TileCoords) => {
    setCoords(newCoords);
    setInputX(newCoords.x.toString());
    setInputY(newCoords.y.toString());
    updateUrlCoords(newCoords);
  }, []);

  // Handle browser back/forward
  useEffect(() => {
    const handlePopState = () => {
      const urlCoords = getCoordsFromUrl();
      setCoords(urlCoords);
      setInputX(urlCoords.x.toString());
      setInputY(urlCoords.y.toString());
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Handle left/right arrow keys to navigate tiles
  useEffect(() => {
    if (availableTiles.length === 0) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if user is typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();

        // Find current index
        const currentIndex = availableTiles.findIndex(
          ([x, y]) => x === coords.x && y === coords.y,
        );

        let newIndex: number;
        if (e.key === "ArrowLeft") {
          // Go to previous (or wrap to end)
          newIndex =
            currentIndex <= 0 ? availableTiles.length - 1 : currentIndex - 1;
        } else {
          // Go to next (or wrap to start)
          newIndex =
            currentIndex >= availableTiles.length - 1 ? 0 : currentIndex + 1;
        }

        const [newX, newY] = availableTiles[newIndex];
        handleCoordsChange({ x: newX, y: newY });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [availableTiles, coords, handleCoordsChange]);

  // Navigate to coordinates from input fields
  const handleGoToCoords = () => {
    const x = parseInt(inputX, 10);
    const y = parseInt(inputY, 10);
    if (!Number.isNaN(x) && !Number.isNaN(y)) {
      handleCoordsChange({ x, y });
    }
  };

  // Handle Enter key in input fields
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleGoToCoords();
    }
  };

  // Jump to a specific tile from the dropdown
  const handleTileSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (value) {
      const [x, y] = value.split(",").map(Number);
      handleCoordsChange({ x, y });
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>
            <span className="emoji">⛄</span> Snow Mode Demo
          </h1>
          <p className="subtitle">
            Isometric NYC • Tile ({coords.x}, {coords.y})
          </p>
        </div>
        <nav className="header-nav">
          <div className="coord-inputs">
            <label className="coord-label">
              <span>X</span>
              <input
                type="number"
                className="coord-input"
                value={inputX}
                onChange={(e) => setInputX(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            </label>
            <label className="coord-label">
              <span>Y</span>
              <input
                type="number"
                className="coord-input"
                value={inputY}
                onChange={(e) => setInputY(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            </label>
            <button className="go-btn" onClick={handleGoToCoords}>
              Go
            </button>
          </div>
          {availableTiles.length > 0 && (
            <select
              className="tile-select"
              value={`${coords.x},${coords.y}`}
              onChange={handleTileSelect}
            >
              {availableTiles.map(([x, y]) => (
                <option key={`${x},${y}`} value={`${x},${y}`}>
                  ({x}, {y})
                </option>
              ))}
            </select>
          )}
        </nav>
      </header>

      <main className="app-main">
        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner" />
            <p>Loading tiles...</p>
          </div>
        )}

        {error && (
          <div className="error-overlay">
            <p className="error-message">⚠️ {error}</p>
            <p className="error-hint">
              Try different coordinates or check that the backend is running.
            </p>
          </div>
        )}

        {!loading && !error && (
          <div className="comparison-container">
            <div className="tile-panel day-panel">
              <div className="panel-label">
                <span className="label-icon">🏙️</span>
                <span>City</span>
              </div>
              {inputUrl ? (
                <img
                  src={inputUrl}
                  alt={`Daytime tile at (${coords.x}, ${coords.y})`}
                  className="tile-image"
                />
              ) : (
                <div className="no-image">
                  <p>No input image</p>
                </div>
              )}
            </div>

            <div className="tile-panel night-panel">
              <div className="panel-label">
                <span className="label-icon">⛄</span>
                <span>Snow</span>
              </div>
              {generationUrl ? (
                <img
                  src={generationUrl}
                  alt={`Nighttime tile at (${coords.x}, ${coords.y})`}
                  className="tile-image"
                />
              ) : (
                <div className="no-image">
                  <p>No generation yet</p>
                  <p className="no-image-hint">
                    Run generate__layer_snow.py to create
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
