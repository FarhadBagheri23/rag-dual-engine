import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import Sidebar from "./components/Sidebar";
import { Menu, Spinner } from "./components/icons";
import AdminView from "./pages/Admin";
import Login from "./pages/Login";
import SearchView from "./pages/Search";

export default function App() {
  // undefined while the stored token is being checked, null once we know there
  // is no session. Without the third state the login screen flashes on every
  // reload for users who are in fact signed in.
  const [user, setUser] = useState(undefined);
  const [tab, setTab] = useState("search");
  const [status, setStatus] = useState(null);
  const [docs, setDocs] = useState([]);
  const [models, setModels] = useState([]);
  const [convs, setConvs] = useState([]);
  // Two pieces of state, not one, because they answer different questions.
  // `active` is a thread to *restore* and changes only when the user picks one
  // — the search view watches it and reloads. `activeId` is only which row is
  // highlighted, and moves on every save. Collapsing them would make saving a
  // search re-trigger the restore that produced it.
  const [active, setActive] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [drawer, setDrawer] = useState(false);

  const signOut = useCallback(() => {
    api.setTokens(null);
    setUser(null);
    setConvs([]);
    setActive(null);
    setActiveId(null);
    setTab("search");
  }, []);

  // A dead refresh token surfaces from inside api.js, mid-request. This is how
  // it gets back out to the one place that can act on it.
  useEffect(() => api.onSessionExpired(signOut), [signOut]);

  useEffect(() => {
    if (!api.getTokens()) return setUser(null);
    api.me().then(setUser).catch(signOut);
  }, [signOut]);

  const reloadDocs = useCallback(() => {
    api.listDocuments().then(setDocs).catch(() => setDocs([]));
  }, []);

  const reloadConvs = useCallback(() => {
    api.listConversations().then(setConvs).catch(() => setConvs([]));
  }, []);

  // Stable identity, and it has to be. The search view keys its "restore a
  // thread" effect on this, so an inline arrow — a new function every App
  // render — would re-run the restore each time the sidebar reloaded, and
  // clear the results of the very search that triggered the reload.
  const handleSaved = useCallback(
    (conv) => {
      setActiveId(conv.id);
      reloadConvs();
    },
    [reloadConvs]
  );

  useEffect(() => {
    if (!user) return;
    api
      .health()
      .then(setStatus)
      .catch((e) => setStatus({ status: "unreachable", error: e.message }));
    api.listModels().then((r) => setModels(r.models)).catch(() => setModels([]));
    reloadDocs();
    reloadConvs();
  }, [user, reloadDocs, reloadConvs]);

  // Escape closes the mobile drawer — the convention, and the only way out for
  // a keyboard user who opened it.
  useEffect(() => {
    if (!drawer) return;
    const onKey = (e) => e.key === "Escape" && setDrawer(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawer]);

  async function openConversation(conv) {
    setDrawer(false);
    setTab("search");
    try {
      setActive(await api.getConversation(conv.id));
      setActiveId(conv.id);
    } catch {
      reloadConvs(); // deleted in another tab; the list was stale
    }
  }

  function newConversation() {
    setActive(null);
    setActiveId(null);
    setDrawer(false);
    setTab("search");
  }

  async function deleteConversation(conv) {
    await api.deleteConversation(conv.id).catch(() => {});
    if (activeId === conv.id) {
      setActive(null);
      setActiveId(null);
    }
    reloadConvs();
  }

  if (user === undefined) {
    return (
      <div className="grid min-h-screen place-items-center">
        <Spinner className="h-6 w-6 text-accent" />
        <span className="sr-only">Checking your session</span>
      </div>
    );
  }

  if (user === null) return <Login onAuthenticated={setUser} />;

  const online = status?.status === "ok";
  const chunks = docs.reduce((n, d) => n + d.n_chunks, 0);
  const isAdmin = user.role === "admin";
  // The tab is only hidden, so the real gate is the backend's get_current_admin
  // on the write routes. This just stops a client seeing a door they cannot open.
  const tabs = [
    { id: "search", label: "Search" },
    ...(isAdmin ? [{ id: "admin", label: "Admin" }] : []),
  ];

  return (
    <div className="min-h-screen md:pl-[17rem]">
      <Sidebar
        conversations={convs}
        activeId={activeId}
        onOpen={openConversation}
        onNew={newConversation}
        onDelete={deleteConversation}
        user={user}
        onSignOut={signOut}
        open={drawer}
        onClose={() => setDrawer(false)}
      />

      <header className="sticky top-0 z-10 border-b border-line bg-bg">
        <div className="mx-auto flex max-w-5xl items-center gap-4 px-6 py-4 sm:gap-8">
          <button
            onClick={() => setDrawer(true)}
            aria-label="Open history"
            aria-expanded={drawer}
            className="-ml-2 grid h-9 w-9 shrink-0 cursor-pointer place-items-center rounded-lg text-dim hover:bg-muted hover:text-fg md:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="min-w-0">
            <h1 className="truncate font-semibold leading-none">
              Dual-Engine Search
            </h1>
            <p className="mt-1 hidden text-xs text-dim sm:block">
              lexical &amp; semantic retrieval over one corpus
            </p>
          </div>

          <nav aria-label="Views" className="flex gap-1">
            {tabs.map((t) => (
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

          <div className="ml-auto hidden items-center gap-4 font-mono text-xs text-dim lg:flex">
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

      <main className="mx-auto w-full max-w-5xl px-6 py-8">
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
          <SearchView
            corpusSize={chunks}
            docCount={docs.length}
            models={models}
            conversation={active}
            onSaved={handleSaved}
          />
        ) : (
          <AdminView docs={docs} reload={reloadDocs} />
        )}
      </main>
    </div>
  );
}
