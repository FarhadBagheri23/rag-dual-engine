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

export const Send = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M4 12 20 4l-8 16-2.2-6.2z" />
  </svg>
);

export const Plus = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const User = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="12" cy="8" r="3.5" />
    <path d="M5 20a7 7 0 0 1 14 0" />
  </svg>
);

export const Logout = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M15 17v1a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v1" />
    <path d="M10 12h10m0 0-3-3m3 3-3 3" />
  </svg>
);

export const Menu = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </svg>
);

export const Close = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

export const Lock = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <rect x="4" y="10" width="16" height="11" rx="2" />
    <path d="M8 10V7a4 4 0 0 1 8 0v3" />
  </svg>
);

export const Mail = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <path d="m3 7 9 6 9-6" />
  </svg>
);

export const Eye = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6-10-6-10-6Z" />
    <circle cx="12" cy="12" r="2.5" />
  </svg>
);

export const EyeOff = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M10.6 6.2A9.6 9.6 0 0 1 12 6c6.4 0 10 6 10 6a17 17 0 0 1-3.1 3.6M6.2 8.4A17 17 0 0 0 2 12s3.6 6 10 6a9.4 9.4 0 0 0 3.9-.8" />
    <path d="M3 3l18 18" />
  </svg>
);

export const History = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M3.5 9A9 9 0 1 1 3 12.5" />
    <path d="M3 4v5h5M12 7.5V12l3 2" />
  </svg>
);

// Meridian + two latitude ellipses, not a filled globe: at 16-20px a filled
// glyph turns into a dot, and the wire globe is what every browser and search
// product uses for "the open web", so it needs no learning.
export const Globe = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18" />
    <path d="M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18" />
  </svg>
);

// An arrow leaving a box — the standard "opens elsewhere" mark. Paired with
// target="_blank" so the icon and the behaviour agree.
export const External = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p}>
    <path d="M14 4h6v6" />
    <path d="m20 4-8.5 8.5" />
    <path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
  </svg>
);

export const Spinner = (p) => (
  <svg viewBox="0 0 24 24" {...base} {...p} className={`animate-spin ${p.className || ""}`}>
    <path d="M12 3a9 9 0 1 0 9 9" />
  </svg>
);
