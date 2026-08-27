// Inline SVG, never emoji — emoji render differently per platform and carry no
// accessible name. Decorative icons are aria-hidden; the label lives in text.
const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

export const Search = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </svg>
);

export const Upload = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M12 15V3m0 0L8 7m4-4 4 4" />
    <path d="M3 15v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4" />
  </svg>
);

export const Trash = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M4 7h16M10 11v6M14 11v6" />
    <path d="M5 7l1 13a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-13M9 7V4h6v3" />
  </svg>
);

export const Doc = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
  </svg>
);

export const Sparkle = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" />
  </svg>
);

export const Check = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="m5 13 4 4L19 7" />
  </svg>
);

export const Alert = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v6M12 16.5v.5" />
  </svg>
);

export const Spinner = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p} className={`animate-spin ${p.className || ""}`}>
    <path d="M12 3a9 9 0 1 0 9 9" />
  </svg>
);
