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
        // Two more hues, for one job: telling VSM / BM25 / RAG apart at a
        // glance down the history sidebar. Every tag also carries its name in
        // text, so colour reinforces the label rather than being the label —
        // the list still reads correctly in greyscale or with a colour vision
        // deficiency. Both cleared 4.5:1 on #1B2336.
        vsm: "#7DD3FC",
        bm25: "#FCD34D",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
