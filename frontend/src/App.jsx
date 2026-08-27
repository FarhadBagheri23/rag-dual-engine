import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import AdminView from "./pages/Admin";
import SearchView from "./pages/Search";

const TABS = [
  { id: "search", label: "Search" },
  { id: "admin", label: "Admin" },
];

export default function App() {
  const [tab, setTab] = useState("search");
  const [status, setStatus] = useState(null);
  const [docs, setDocs] = useState([]);
  const [models, setModels] = useState([]);

  const reload = useCallback(() => {
    api.listDocuments().then(setDocs).catch(() => setDocs([]));
  }, []);

  useEffect(() => {
    api
      .health()
      .then(setStatus)
      .catch((e) => setStatus({ status: "unreachable", error: e.message }));
    api.listModels().then((r) => setModels(r.models)).catch(() => setModels([]));
    reload();
  }, [reload]);

  const online = status?.status === "ok";
  const chunks = docs.reduce((n, d) => n + d.n_chunks, 0);

  return (
    <div className="min-h-screen">
      <header className="border-b border-line">
        <div className="mx-auto flex max-w-5xl items-center gap-8 px-6 py-4">
          <div>
            <h1 className="font-semibold leading-none">Dual-Engine Search</h1>
            <p className="mt-1 text-xs text-dim">
              lexical &amp; semantic retrieval over one corpus
            </p>
          </div>

          <nav aria-label="Views" className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                className={`seg ${tab === t.id ? "seg-on" : "seg-off"}`}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-4 font-mono text-xs text-dim">
            <span>
              {docs.length} docs &middot; {chunks} chunks
            </span>
            <span className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  online ? "bg-accent" : "bg-danger"
                }`}
              />
              {online
                ? status.llm_configured
                  ? status.llm_model
                  : "no llm key"
                : "api offline"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {!online && status && (
          <div className="card mb-5 border-danger/50 p-4 text-sm">
            Backend unreachable on port 8000. Start it with{" "}
            <code className="font-mono text-dim">
              uvicorn app.main:app --reload
            </code>
            .
          </div>
        )}

        {tab === "search" ? (
          <SearchView corpusSize={chunks} models={models} />
        ) : (
          <AdminView docs={docs} reload={reload} />
        )}
      </main>
    </div>
  );
}
