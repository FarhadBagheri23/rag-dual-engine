import { useEffect, useState } from "react";
import { health } from "./api";

// ponytail: useState tab instead of react-router — two views, no deep links yet.
export default function App() {
  const [tab, setTab] = useState("search");
  const [api, setApi] = useState(null);

  useEffect(() => {
    health()
      .then(setApi)
      .catch((e) => setApi({ status: "unreachable", error: e.message }));
  }, []);

  const online = api?.status === "ok";

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-4">
          <h1 className="font-semibold">Dual-Engine Search</h1>
          <nav className="flex gap-1">
            {["search", "admin"].map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-md px-3 py-1.5 text-sm capitalize ${
                  tab === t
                    ? "bg-slate-900 text-white"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {t}
              </button>
            ))}
          </nav>
          <span className="ml-auto flex items-center gap-2 text-sm text-slate-500">
            <span
              className={`h-2 w-2 rounded-full ${
                online ? "bg-emerald-500" : "bg-red-500"
              }`}
            />
            {online ? `api · ${api.llm_model}` : "api offline"}
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        <p className="text-slate-500">
          Phase 0 — skeleton. {tab === "search" ? "Search" : "Admin"} view lands
          in phase 5.
        </p>
      </main>
    </div>
  );
}
