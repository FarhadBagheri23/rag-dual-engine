import { useState } from "react";
import * as api from "../api";
import { Alert, Eye, EyeOff, Lock, Mail, Search, Sparkle, Spinner } from "../components/icons";

// Both panels are the same form with a different verb, so they are one
// component with a mode flag rather than two files that drift apart.
const COPY = {
  login: {
    heading: "Welcome back",
    sub: "Sign in to search the corpus and pick up your history.",
    submit: "Sign in",
    swap: "New here? Create an account",
    autocomplete: "current-password",
  },
  register: {
    heading: "Create an account",
    sub: "Client accounts can use every engine and keep their own history.",
    submit: "Create account",
    swap: "Already have an account? Sign in",
    autocomplete: "new-password",
  },
};

export default function Login({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [reveal, setReveal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const copy = COPY[mode];

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const call = mode === "login" ? api.login : api.register;
      const tokens = await call(email.trim(), password);
      api.setTokens(tokens);
      onAuthenticated(tokens.user);
    } catch (err) {
      setError(err.message);
      setBusy(false); // stays busy on success — the app is about to unmount this
    }
  }

  return (
    // grid-rows-[1fr], because min-h-screen alone only sizes the *container*:
    // the single implicit row still sizes to its content, leaving the identity
    // panel floating short of the fold.
    <div className="grid min-h-screen grid-rows-[1fr] lg:grid-cols-2">
      {/* Identity panel. Hidden below lg rather than stacked: on a phone it
          would push the form under the fold, and the form is the whole job. */}
      <aside className="relative hidden overflow-hidden border-r border-line bg-card lg:flex lg:flex-col lg:justify-between lg:p-12">
        <div
          aria-hidden
          className="pointer-events-none absolute -right-24 -top-24 h-96 w-96 rounded-full bg-accent/10 blur-3xl"
        />
        <div className="relative">
          <span className="grid h-11 w-11 place-items-center rounded-xl border border-accent/40 bg-accent/10">
            <Search className="h-5 w-5 text-accent" />
          </span>
          <h1 className="mt-6 text-3xl font-semibold leading-tight">
            Dual-Engine Search
          </h1>
          <p className="mt-3 max-w-sm text-dim">
            Lexical and semantic retrieval over one corpus — the vector space
            model, Okapi BM25, and retrieval-augmented generation, each answering
            the same question its own way.
          </p>
        </div>

        <ul className="relative mt-10 space-y-4 text-sm">
          {[
            ["VSM", "text-vsm", "tf-idf cosine ranking with Rocchio feedback"],
            ["BM25", "text-bm25", "probabilistic ranking with term saturation"],
            ["RAG", "text-accent", "grounded answers, cited back to the chunk"],
          ].map(([tag, tone, what]) => (
            <li key={tag} className="flex items-baseline gap-3">
              <span
                className={`w-14 shrink-0 rounded border border-line bg-muted px-1.5 py-0.5 text-center font-mono text-[0.7rem] ${tone}`}
              >
                {tag}
              </span>
              <span className="text-dim">{what}</span>
            </li>
          ))}
        </ul>

        <p className="relative mt-10 font-mono text-xs text-dim">
          Modern Information Retrieval · final project
        </p>
      </aside>

      <main className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <span className="grid h-10 w-10 place-items-center rounded-xl border border-accent/40 bg-accent/10 lg:hidden">
            <Sparkle className="h-5 w-5 text-accent" />
          </span>

          <h2 className="mt-5 text-2xl font-semibold lg:mt-0">{copy.heading}</h2>
          <p className="mt-1.5 text-sm text-dim">{copy.sub}</p>

          <form onSubmit={submit} className="mt-7 space-y-4">
            <div>
              <label htmlFor="email" className="mb-1.5 block text-sm font-medium">
                Email
              </label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-dim" />
                <input
                  id="email"
                  type="email"
                  required
                  autoComplete="email"
                  autoFocus
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full rounded-lg border border-line bg-card py-2.5 pl-10 pr-3 placeholder:text-dim"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="mb-1.5 block text-sm font-medium">
                Password
              </label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-dim" />
                <input
                  id="password"
                  type={reveal ? "text" : "password"}
                  required
                  minLength={8}
                  autoComplete={copy.autocomplete}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="At least 8 characters"
                  aria-describedby={mode === "register" ? "pw-hint" : undefined}
                  className="w-full rounded-lg border border-line bg-card py-2.5 pl-10 pr-11 placeholder:text-dim"
                />
                {/* Typing a password you cannot see is the most common reason a
                    correct one gets rejected. aria-pressed, not a label swap,
                    so a screen reader hears the state rather than a guess. */}
                <button
                  type="button"
                  onClick={() => setReveal((r) => !r)}
                  aria-pressed={reveal}
                  aria-label="Show password"
                  className="absolute right-1.5 top-1/2 grid h-8 w-8 -translate-y-1/2 cursor-pointer place-items-center rounded-md text-dim transition-colors duration-150 hover:bg-muted hover:text-fg"
                >
                  {reveal ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {mode === "register" && (
                <p id="pw-hint" className="mt-1.5 text-xs text-dim">
                  8–72 characters. Stored as a salted bcrypt hash, never as text.
                </p>
              )}
            </div>

            {error && (
              <div
                role="alert"
                className="card flex items-start gap-2.5 border-danger/50 p-3 text-sm"
              >
                <Alert className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={busy}
              className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg bg-accent py-2.5 font-medium text-accent-ink transition-colors duration-150 hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy && <Spinner className="h-4 w-4" />}
              {copy.submit}
            </button>
          </form>

          <button
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError(null);
            }}
            className="mt-5 w-full cursor-pointer text-center text-sm text-dim transition-colors duration-150 hover:text-fg"
          >
            {copy.swap}
          </button>
        </div>
      </main>
    </div>
  );
}
