import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import { Alert, Check, Doc, Send, Sparkle, Spinner } from "./icons";

/** What each backend stage is called in the UI. The names are the pipeline's
 *  real steps — rag.stream emits one before each, so nothing here is on a
 *  timer pretending to be progress. */
const STAGES = {
  rewriting: "Working out what you mean",
  embedding: "Embedding the question",
  retrieving: "Searching the vector index",
  reading: "Reading the passages",
  writing: "Writing the answer",
};

const HISTORY_TURNS = 6; // what the backend keeps; sending more is wasted bytes

const SUGGESTIONS = [
  "How does BM25 differ from plain tf-idf?",
  "What stops a language model from making things up?",
  "Why normalise document length?",
];

const CITE = /(\[Doc\s*\d+\])/gi;

/** Render [Doc 1] markers in the answer as inline badges. Same alternating
 *  split as Results.jsx: with one capture group, odd indices are the matches. */
function withCitations(text) {
  return text.split(CITE).map((part, i) =>
    i % 2 === 1 ? (
      <span
        key={i}
        className="mx-0.5 whitespace-nowrap rounded bg-accent/15 px-1.5 py-0.5 align-baseline font-mono text-[0.7rem] text-accent"
      >
        {part}
      </span>
    ) : (
      part
    )
  );
}

function Stages({ stages, elapsed, collapsed }) {
  const list = (
    <ul className="space-y-2">
      {stages.map((s) => (
        <li key={s.name} className="flex items-start gap-2.5 text-sm">
          <span className="mt-0.5 h-4 w-4 shrink-0">
            {s.done ? (
              <Check className="h-4 w-4 text-accent" />
            ) : (
              <Spinner className="h-4 w-4 text-accent" />
            )}
          </span>
          <span className="min-w-0">
            <span className={s.done ? "text-dim" : "text-fg"}>
              {STAGES[s.name] || s.name}
            </span>
            {s.query && (
              // The standalone question the follow-up resolved to. Shown
              // because it is what actually got embedded — if retrieval goes
              // wrong, this is the first place to look.
              <span dir="auto" className="mt-1 block text-xs italic text-accent">
                “{s.query}”
              </span>
            )}
            {s.detail && (
              <span className="mt-1 flex flex-wrap gap-1.5">
                {/* Keyed by position, not title: retrieval routinely returns
                    several chunks of the *same* document, so titles collide and
                    React drops the duplicates. The list is static once the
                    stage is recorded, so the index is a stable key. */}
                {s.detail.map((d, di) => (
                  <span
                    key={di}
                    dir="auto"
                    className="rounded border border-line bg-muted px-1.5 py-0.5 text-xs text-dim"
                  >
                    {d}
                  </span>
                ))}
              </span>
            )}
          </span>
        </li>
      ))}
    </ul>
  );

  // Once the answer starts arriving the steps stop being the story, so they
  // fold away behind a one-line summary rather than pushing the text down.
  if (!collapsed) return <div className="py-1">{list}</div>;
  return (
    <details className="mb-3 border-b border-line pb-3">
      <summary className="cursor-pointer text-xs uppercase tracking-wide text-dim hover:text-fg">
        Worked for {(elapsed / 1000).toFixed(1)}s · {stages.length} steps
      </summary>
      <div className="mt-3">{list}</div>
    </details>
  );
}

function Sources({ citations, hits }) {
  const cited = new Set(citations.map((c) => c.doc_number));
  return (
    <div className="mt-3 border-t border-line pt-3">
      {citations.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {citations.map((c) => (
            <span
              key={c.doc_number}
              className="inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/10 px-2 py-1 text-xs"
            >
              <Doc className="h-3.5 w-3.5 shrink-0 text-accent" />
              <span className="font-mono text-accent">[Doc {c.doc_number}]</span>
              <span dir="auto">{c.title}</span>
            </span>
          ))}
        </div>
      )}
      {hits.length > 0 && (
        <details className={citations.length ? "mt-3" : ""}>
          <summary className="cursor-pointer text-xs uppercase tracking-wide text-dim hover:text-fg">
            Retrieved passages ({hits.length})
          </summary>
          <ul className="mt-2 space-y-1.5">
            {hits.map((h) => (
              <li key={h.chunk_id} className="flex items-baseline gap-2.5 text-sm">
                <span
                  className={`font-mono text-xs ${
                    cited.has(h.doc_number) ? "text-accent" : "text-dim"
                  }`}
                >
                  [Doc {h.doc_number}]
                </span>
                <span dir="auto" className="flex-1">
                  {h.title}
                </span>
                <span className="font-mono text-xs text-dim">
                  {h.score.toFixed(4)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function Bubble({ msg }) {
  if (msg.role === "user") {
    return (
      <li className="flex justify-end">
        <div
          dir="auto"
          className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-accent-ink"
        >
          {msg.text}
        </div>
      </li>
    );
  }

  const working = msg.stages.length > 0 && !msg.done;
  const hasText = msg.text.length > 0;

  return (
    // items-start, not Telegram's bottom-anchored avatar: a grounded answer
    // runs long, and an icon floating beside the last line reads as detached.
    <li className="flex items-start gap-2">
      <span className="mt-1 grid h-7 w-7 shrink-0 place-items-center rounded-full border border-accent/40 bg-accent/10">
        <Sparkle className="h-4 w-4 text-accent" />
      </span>

      <div className="card max-w-[85%] rounded-2xl rounded-bl-md px-4 py-3">
        {msg.stages.length > 0 && (
          <Stages
            stages={msg.stages}
            elapsed={msg.took_ms ?? msg.elapsed}
            collapsed={hasText || msg.done}
          />
        )}

        {hasText && (
          <p dir="auto" className="whitespace-pre-wrap leading-relaxed">
            {withCitations(msg.text)}
            {working && (
              <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-accent" />
            )}
          </p>
        )}

        {msg.note && (
          <p className="flex items-start gap-2 text-sm text-dim">
            <Alert className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{msg.note}</span>
          </p>
        )}

        {msg.coverage?.length > 0 && (
          <ul className="mt-3 flex flex-wrap gap-2">
            {msg.coverage.map((t) => (
              <li
                key={t}
                dir="auto"
                className="rounded-md border border-line bg-muted px-2 py-1 text-xs"
              >
                {t}
              </li>
            ))}
          </ul>
        )}

        {msg.error && (
          <p className="flex items-start gap-2 text-sm text-danger">
            <Alert className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{msg.error}</span>
          </p>
        )}

        {msg.done && (msg.citations.length > 0 || msg.hits.length > 0) && (
          <Sources citations={msg.citations} hits={msg.hits} />
        )}

        {msg.done && msg.model && (
          <div className="mt-2 text-right font-mono text-[0.7rem] text-dim">
            {msg.model} · {(msg.took_ms / 1000).toFixed(1)}s
          </div>
        )}
      </div>
    </li>
  );
}

/** Stored shape → the shape a Bubble renders.
 *
 *  A restored turn is `done` with no stages: the steps described work that
 *  happened once, live, and replaying them as if they were happening again
 *  would be theatre. The answer and its sources are the parts worth keeping. */
function restore(stored) {
  if (stored.role === "user") return { role: "user", text: stored.content };
  const meta = stored.meta ?? {};
  return {
    role: "assistant",
    text: stored.content,
    stages: [],
    hits: meta.hits ?? [],
    citations: meta.citations ?? [],
    model: meta.model,
    took_ms: meta.took_ms,
    note: meta.note,
    coverage: meta.coverage,
    done: true,
  };
}

/** Bubble → what goes in the database. RAG answers are stored in full, unlike
 *  lexical hits: regenerating one costs an LLM call and would not come back
 *  the same anyway, so a replay has to be a replay. */
function store(msg) {
  if (msg.role === "user") return { role: "user", content: msg.text };
  return {
    role: "assistant",
    content: msg.text,
    meta: {
      citations: msg.citations ?? [],
      hits: msg.hits ?? [],
      model: msg.model,
      took_ms: msg.took_ms,
      note: msg.note,
      coverage: msg.coverage,
    },
  };
}

export default function Chat({ model, corpusSize, conversation, onSaved }) {
  const [messages, setMessages] = useState(() =>
    (conversation?.messages ?? []).map(restore)
  );
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const pinned = useRef(true); // only autoscroll if the user hasn't scrolled up
  const box = useRef(null);
  // The thread this chat is appending to. A ref, not state: it changes as a
  // side effect of saving and nothing renders from it, so re-rendering on it
  // would be a wasted pass.
  const convId = useRef(conversation?.id ?? null);

  // The page is the scroll container, so following the answer means moving the
  // window. Reading back is sacred: once the user scrolls up mid-generation,
  // stop yanking them to the bottom until they come back down themselves.
  useEffect(() => {
    const onScroll = () => {
      const room =
        document.documentElement.scrollHeight -
        window.scrollY -
        window.innerHeight;
      pinned.current = room < 120;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (pinned.current) {
      window.scrollTo({ top: document.documentElement.scrollHeight });
    }
  });

  async function send(question = draft) {
    const q = question.trim();
    if (!q || busy) return;
    setDraft("");
    setBusy(true);
    pinned.current = true;

    const started = performance.now();
    const reply = {
      role: "assistant",
      text: "",
      stages: [],
      hits: [],
      citations: [],
      done: false,
      elapsed: 0,
    };
    // Prior turns, so the backend can resolve a follow-up like "just it?"
    // against them (RAG slides s24, query rewriting). Assistant turns that
    // produced no text — a rejected query, an error — carry no meaning to
    // resolve against, so they are left out.
    const history = messages
      .filter((m) => m.text)
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.text }));

    setMessages((m) => [...m, { role: "user", text: q }, reply]);

    // The finished reply, captured as it is built. `messages` in this closure
    // is the transcript as it stood before this turn, so the two together are
    // the whole thread to persist — read out of state afterwards it would be
    // one render behind.
    let final = reply;

    // Every update rewrites the last message in place; `patch` keeps that from
    // being spelled out six times below.
    const patch = (fn) =>
      setMessages((m) => {
        final = fn({ ...m[m.length - 1] });
        return [...m.slice(0, -1), final];
      });

    try {
      for await (const event of api.searchStream({ query: q, model, history })) {
        if (event.rewritten) {
          // Attach to the stage that produced it, rather than adding a row.
          patch((r) => ({
            ...r,
            stages: r.stages.map((s, i) =>
              i === r.stages.length - 1 ? { ...s, query: event.rewritten } : s
            ),
          }));
        } else if (event.stage) {
          patch((r) => {
            const stages = r.stages.map((s) => ({ ...s, done: true }));
            stages.push({
              name: event.stage,
              done: false,
              detail: event.hits?.map((h) => h.title),
            });
            return { ...r, stages, elapsed: performance.now() - started };
          });
        } else if (event.delta) {
          patch((r) => ({
            ...r,
            text: r.text + event.delta,
            stages: r.stages.map((s) => ({ ...s, done: true })),
          }));
        } else if (event.done) {
          patch((r) => ({
            ...r,
            ...event.done,
            text: event.done.answer ?? r.text,
            stages: r.stages.map((s) => ({ ...s, done: true })),
            done: true,
          }));
        } else if (event.error) {
          patch((r) => ({ ...r, error: event.error, done: true }));
        }
      }
    } catch (e) {
      patch((r) => ({ ...r, error: e.message, done: true }));
    } finally {
      setBusy(false);
      patch((r) => ({ ...r, done: true }));
      box.current?.focus();

      // A turn that produced no answer — a provider error, a rejected query —
      // is not history worth a sidebar row. Everything else is appended to the
      // thread, which the first save creates.
      if (final.text) {
        api
          .saveConversation({
            id: convId.current,
            engine: "rag",
            messages: [
              ...messages.map(store),
              { role: "user", content: q },
              store({ ...final, role: "assistant" }),
            ],
          })
          .then((conv) => {
            convId.current = conv.id;
            onSaved?.(conv);
          })
          // Losing the history entry must not look like a failed answer — the
          // answer is on screen and correct either way.
          .catch(() => {});
      }
    }
  }

  return (
    <div>
      {/* Enough height that the sticky composer starts at the bottom of the
          viewport rather than floating mid-page on a short conversation. The
          measured chrome above and below is ~20.4rem; 22rem leaves slack, so
          an empty chat never produces a scrollbar. Once the transcript is
          taller than this the value stops mattering. */}
      <div className="min-h-[calc(100vh-22rem)]">
        {messages.length === 0 ? (
          <div className="flex min-h-[inherit] flex-col items-center justify-center text-center">
            <span className="grid h-12 w-12 place-items-center rounded-full border border-accent/40 bg-accent/10">
              <Sparkle className="h-6 w-6 text-accent" />
            </span>
            <p className="mt-4 font-medium">
              Ask the {corpusSize} indexed chunks anything
            </p>
            <p className="mt-1.5 max-w-md text-sm text-dim">
              Answers are written only from passages retrieved out of your
              corpus, and every claim is cited back to the chunk it came from.
            </p>
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="cursor-pointer rounded-full border border-line px-3.5 py-1.5 text-sm text-dim transition-colors duration-150 hover:border-edge hover:text-fg"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <ul className="space-y-4 pb-4" aria-live="polite">
            {messages.map((m, i) => (
              <Bubble key={i} msg={m} />
            ))}
          </ul>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        // Pinned to the bottom of the viewport; the transcript scrolls under
        // it. -mx-6/px-6 lets the backdrop span the full gutter so text does
        // not show through at the edges as it passes behind.
        className="sticky bottom-0 -mx-6 mt-3 flex items-end gap-2 border-t border-line bg-bg px-6 pb-6 pt-3"
      >
        <label htmlFor="ask" className="sr-only">
          Ask a question about the corpus
        </label>
        {/* Send sits left of the box. It stays first in the DOM too, so the
            tab order matches the visual order for keyboard and screen-reader
            users rather than only looking right. */}
        <button
          type="submit"
          disabled={busy || !draft.trim()}
          aria-label="Send"
          className="grid h-12 w-12 shrink-0 cursor-pointer place-items-center rounded-full bg-accent text-accent-ink transition-colors duration-150 hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? <Spinner className="h-5 w-5" /> : <Send className="h-5 w-5" />}
        </button>
        <textarea
          id="ask"
          ref={box}
          dir="auto"
          rows={1}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter breaks the line — the chat convention.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask a question…"
          className="max-h-40 flex-1 resize-none rounded-2xl border border-line bg-card px-4 py-3 leading-relaxed placeholder:text-dim"
        />
      </form>
    </div>
  );
}
