// ponytail: plain fetch, no axios/react-query. Five endpoints.
async function json(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  // FastAPI puts the useful message in `detail` — a restricted API key or an
  // unparseable file explains itself there, so surface it instead of a status.
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

export const health = () => json("/api/health");
export const listModels = () => json("/api/models");
export const listDocuments = () => json("/api/documents");

export const uploadDocument = (file) => {
  const form = new FormData();
  form.append("file", file);
  return json("/api/documents", { method: "POST", body: form });
};

export const deleteDocument = (id) =>
  json(`/api/documents/${id}`, { method: "DELETE" });

export const search = (body) =>
  json("/api/search", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
