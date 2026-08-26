import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// ponytail: proxy /api to the backend so the frontend needs no API base URL,
// no CORS dance in dev, and no VITE_ env var. Same-origin in prod via nginx.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
