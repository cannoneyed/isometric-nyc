import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  // Load env file from project root (two levels up from src/app/).
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, resolve(__dirname, "../.."), "");

  return {
    // Use /isometric-nyc/ base path for production (GitHub Pages)
    base: mode === "production" ? "/isometric-nyc/" : "/",
    plugins: [react()],
    resolve: {
      alias: {
        "@": resolve(__dirname, "src"),
      },
    },
    server: {
      port: 3000,
      open: true,
    },
    define: {
      // R2 URL for DZI tiles (served directly from R2 bucket)
      // Production: use R2 base URL (export dir is appended via config.ts)
      // Dev: use local public folder (empty string = same origin)
      __TILES_BASE_URL__: JSON.stringify(
        mode === "production"
          ? "https://isometric-nyc-tiles.cannoneyed.com"
          : ""
      ),
      // MAP_ID determines which subdirectory under public/dzi/ to load
      // Set via: MAP_ID=tiny-nyc in .env file or MAP_ID=tiny-nyc bun run dev
      __MAP_ID__: JSON.stringify(env.MAP_ID || ""),
      // LOCAL_R2 forces use of R2 tiles in dev mode
      // Set via: LOCAL_R2=true in .env file or LOCAL_R2=true bun run dev
      __LOCAL_R2__: JSON.stringify(env.LOCAL_R2 === "true"),
      // USE_R2_NYC fetches from R2 dzi/ directly (no map id subdir)
      // Set via: USE_R2_NYC=true in .env file or USE_R2_NYC=true bun run dev
      __USE_R2_NYC__: JSON.stringify(env.USE_R2_NYC === "true"),
    },
  };
});
