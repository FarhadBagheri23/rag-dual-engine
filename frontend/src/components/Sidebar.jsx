import { Close, History, Logout, Plus, Trash, User } from "./icons";

/** Engine → the colour its tag is drawn in. The tag always carries the engine
 *  *name* too, so this reinforces the label rather than being it — the list
 *  still reads in greyscale, and to anyone who cannot separate the hues. */
const TAG_TONE = {
  vsm: "text-vsm",
  bm25: "text-bm25",
  rag: "text-accent",
};

const DAY = 86_400_000;

/** Which heading a thread files under. Compared in local midnights, not raw
 *  timestamps: something from 23:50 last night is "Yesterday" at 00:10 today,
 *  and a plain hours-ago subtraction would still call it "Today". */
function bucket(iso) {
  const midnight = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const days = Math.round((midnight(new Date()) - midnight(new Date(iso))) / DAY);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return "Older";
}

/** Group an already-sorted list, keeping its order. A Map because insertion
 *  order is the display order — the backend sorted by updated_at DESC, so the
 *  headings come out newest-first without a second sort here. */
function grouped(conversations) {
  const groups = new Map();
  for (const c of conversations) {
    const key = bucket(c.updated_at);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(c);
  }
  return [...groups];
}

function Row({ conv, active, onOpen, onDelete }) {
  return (
    <li className="group relative">
      <button
        onClick={() => onOpen(conv)}
        aria-current={active ? "page" : undefined}
        className={`flex w-full items-baseline gap-2 rounded-md py-2 pl-2 pr-9 text-left transition-colors duration-150 ${
          active ? "bg-muted" : "hover:bg-muted/60"
        }`}
      >
        <span
          className={`shrink-0 font-mono text-[0.65rem] uppercase ${TAG_TONE[conv.engine]}`}
        >
          {conv.engine}
        </span>
        {/* dir="auto" — a Persian query must read right-to-left in its own row
            without flipping the tag or the list around it. */}
        <span dir="auto" className="min-w-0 flex-1 truncate text-sm">
          {conv.title}
        </span>
      </button>

      {/* Hover reveals it on a pointer; focus-within reveals it on a keyboard,
          so the control is not unreachable by tab. */}
      <button
        onClick={() => onDelete(conv)}
        aria-label={`Delete conversation: ${conv.title}`}
        className="absolute right-1 top-1/2 grid h-7 w-7 -translate-y-1/2 cursor-pointer place-items-center rounded text-dim opacity-0 transition-all duration-150 hover:bg-danger/15 hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
      >
        <Trash className="h-3.5 w-3.5" />
      </button>
    </li>
  );
}

export default function Sidebar({
  conversations,
  activeId,
  onOpen,
  onNew,
  onDelete,
  user,
  onSignOut,
  open,
  onClose,
}) {
  return (
    <>
      {/* Backdrop, mobile only. A button rather than a div so dismissing the
          drawer is not pointer-only. */}
      {open && (
        <button
          onClick={onClose}
          aria-label="Close history"
          className="fixed inset-0 z-20 bg-bg/70 backdrop-blur-sm md:hidden"
        />
      )}

      <div
        className={`fixed inset-y-0 left-0 z-30 flex w-[17rem] flex-col border-r border-line bg-card transition-transform duration-200 md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-2 p-3">
          <button
            onClick={onNew}
            className="flex flex-1 cursor-pointer items-center gap-2 rounded-lg border border-line px-3 py-2 text-sm font-medium transition-colors duration-150 hover:border-edge hover:bg-muted"
          >
            <Plus className="h-4 w-4 text-accent" />
            New search
          </button>
          <button
            onClick={onClose}
            aria-label="Close history"
            className="grid h-9 w-9 shrink-0 cursor-pointer place-items-center rounded-lg text-dim hover:bg-muted hover:text-fg md:hidden"
          >
            <Close className="h-4 w-4" />
          </button>
        </div>

        <nav
          aria-label="Conversation history"
          className="min-h-0 flex-1 overflow-y-auto px-2 pb-2"
        >
          {conversations.length === 0 ? (
            <div className="px-2 py-10 text-center">
              <History className="mx-auto h-6 w-6 text-dim" />
              <p className="mt-3 text-sm text-dim">No history yet</p>
              <p className="mt-1 text-xs text-dim">
                Every search and chat you run is saved here, tagged with the
                engine that answered it.
              </p>
            </div>
          ) : (
            grouped(conversations).map(([heading, items]) => (
              <section key={heading} className="mb-3">
                <h2 className="px-2 pb-1 pt-2 text-[0.7rem] font-medium uppercase tracking-wide text-dim">
                  {heading}
                </h2>
                <ul>
                  {items.map((c) => (
                    <Row
                      key={c.id}
                      conv={c}
                      active={c.id === activeId}
                      onOpen={onOpen}
                      onDelete={onDelete}
                    />
                  ))}
                </ul>
              </section>
            ))
          )}
        </nav>

        <div className="flex items-center gap-2.5 border-t border-line p-3">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-line bg-muted">
            <User className="h-4 w-4 text-dim" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm" title={user.email}>
              {user.email}
            </div>
            <div
              className={`font-mono text-[0.65rem] uppercase ${
                user.role === "admin" ? "text-accent" : "text-dim"
              }`}
            >
              {user.role}
            </div>
          </div>
          <button
            onClick={onSignOut}
            aria-label="Sign out"
            title="Sign out"
            className="grid h-8 w-8 shrink-0 cursor-pointer place-items-center rounded-md text-dim transition-colors duration-150 hover:bg-danger/15 hover:text-danger"
          >
            <Logout className="h-4 w-4" />
          </button>
        </div>
      </div>
    </>
  );
}
