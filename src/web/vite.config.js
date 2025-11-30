import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  // Look for .env files in the project root (two levels up from src/web/)
  envDir: resolve(__dirname, "../.."),
});
