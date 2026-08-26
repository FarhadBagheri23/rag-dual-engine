// ponytail: plain fetch, no axios/react-query. Three endpoints total.
async function json(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export const health = () => json("/api/health");
