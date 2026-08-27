/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0F172A",
        card: "#1B2336",
        muted: "#272F42",
        line: "#2A3550",
        edge: "#475569",
        fg: "#F8FAFC",
        dim: "#94A3B8",
        accent: "#22C55E",
        "accent-ink": "#0F172A",
        danger: "#EF4444",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
