import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => ({
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
    // R2 public URL for tiles (empty string = same origin, for dev)
    __TILES_BASE_URL__: JSON.stringify(
      mode === "production"
        ? "https://pub-4d4013d62ff44fc6b63d825226ec07bd.r2.dev"
        : ""
    ),
  },
}));
