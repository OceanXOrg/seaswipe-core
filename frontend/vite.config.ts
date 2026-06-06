import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// In dev (`npm run dev`) the app runs on :3000 and proxies /api to the
// FastAPI server on :8080. In production it is built to dist/ and served by
// the FastAPI server directly, so no proxy is involved.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
      },
    },
  },
});
