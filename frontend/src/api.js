// ponytail: plain fetch, no axios/react-query.
//
// Tokens live here rather than in React state so that both call sites below —
// `json` and the streaming reader — attach the header from one place. Putting
// them in a context would mean every caller passing a token down, and the one
// caller that forgot would be the one that streams.

const STORE = "mir.auth";

let tokens = JSON.parse(localStorage.getItem(STORE) || "null");
let onExpired = () => {};

export const getTokens = () => tokens;

export function setTokens(next) {
  tokens = next;
  if (next) localStorage.setItem(STORE, JSON.stringify(next));
  else localStorage.removeItem(STORE);
}

/** Called when the refresh token is dead too — the app shows the login screen. */
export function onSessionExpired(fn) {
  onExpired = fn;
}

// Concurrent 401s (the app fires three requests on mount) share one refresh
// rather than each firing their own.
let refreshing = null;

function refreshTokens() {
  refreshing ||= fetch("/api/auth/refresh", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ refresh_token: tokens.refresh_token }),
  })
    .then((r) => (r.ok ? r.json() : null))
    .then((next) => {
      setTokens(next); // null on failure, which clears the dead pair
      return next;
    })
    .catch(() => null)
    .finally(() => {
      refreshing = null;
    });
  return refreshing;
}

/** fetch + bearer token, retried once against a fresh access token.
 *
 *  The 30-minute access token will expire mid-session, and the only place that
 *  is discoverable is a 401 on an ordinary request. Refreshing there and
 *  replaying the request keeps that invisible; only a dead *refresh* token
 *  reaches the user, as a return to the login screen. */
async function authedFetch(path, options = {}) {
  const send = () =>
    fetch(path, {
      ...options,
      headers: {
        ...options.headers,
        ...(tokens ? { Authorization: `Bearer ${tokens.access_token}` } : {}),
      },
    });

  let res = await send();
  if (res.status === 401 && tokens?.refresh_token) {
    if (await refreshTokens()) res = await send();
    else onExpired();
  }
  return res;
}

async function json(path, options) {
  const res = await authedFetch(path, options);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  // FastAPI puts the useful message in `detail` — a restricted API key or an
  // unparseable file explains itself there, so surface it instead of a status.
  if (!res.ok) throw new Error(detailOf(body) || `Request failed (${res.status})`);
  return body;
}

/** 422s arrive as a list of field errors, not a string — "password: String
 *  should have at least 8 characters" beats "[object Object]". */
function detailOf(body) {
  const d = body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => `${e.loc?.[e.loc.length - 1] ?? "field"}: ${e.msg}`)
      .join("; ");
  }
  return null;
}

const post = (path, body) =>
  json(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

// --- auth ---
export const register = (email, password) =>
  post("/api/auth/register", { email, password });
export const login = (email, password) =>
  post("/api/auth/login", { email, password });
export const me = () => json("/api/auth/me");

// --- corpus & search ---
export const health = () => fetch("/api/health").then((r) => r.json());
export const listModels = () => json("/api/models");
export const listDocuments = () => json("/api/documents");

export const uploadDocument = (file, title) => {
  const form = new FormData();
  form.append("file", file);
  // Omitted entirely when blank, so the backend's fallback chain decides.
  if (title) form.append("title", title);
  return json("/api/documents", { method: "POST", body: form });
};

export const deleteDocument = (id) =>
  json(`/api/documents/${id}`, { method: "DELETE" });

export const search = (body) => post("/api/search", body);

// --- conversation history ---
export const listConversations = () => json("/api/conversations");
export const getConversation = (id) => json(`/api/conversations/${id}`);
export const saveConversation = (body) => post("/api/conversations", body);
export const deleteConversation = (id) =>
  json(`/api/conversations/${id}`, { method: "DELETE" });

/** RAG as a stream of events: {stage}, {delta}, {done}, {error}.
 *
 *  An async generator, so the caller writes `for await (const e of ...)` and
 *  the buffering stays here. NDJSON over fetch rather than EventSource —
 *  EventSource is GET-only, cannot send a body, and cannot carry a bearer
 *  header, all three of which this needs. */
export async function* searchStream(body) {
  const res = await authedFetch("/api/search/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...body, engine: "rag" }),
  });
  if (!res.ok) {
    const b = await res.json().catch(() => ({}));
    throw new Error(detailOf(b) || `Request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // a chunk can split a line in half — keep the tail
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
  }
}
