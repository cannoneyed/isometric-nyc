# Final app ship

We want to ship the final app in src/app! For production, we need to remove a few things

- the controls for shaders, scanlines, etc
- the x,y debug text

Make this triggered by a production build flag that can be overridden in the url by adding "?prod=true" to the url in dev mode

## ✅ Completed

Implemented in `src/app/src/App.tsx`:

- Added `shouldShowDebugUI()` function that determines visibility based on:
  - **Production builds**: Debug UI hidden by default
  - **Dev builds**: Debug UI shown by default
  - **URL overrides**:
    - `?prod=true` - Force production mode (hide debug UI)
    - `?debug=true` - Force debug mode (show debug UI)
- Updated conditional rendering of `ControlPanel` and `TileInfo` components to use `showDebugUI`
- Water shader is also disabled when in production mode
