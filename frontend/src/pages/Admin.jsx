import { useRef, useState } from "react";
import * as api from "../api";
import { Alert, Check, Doc, Spinner, Trash, Upload } from "../components/icons";

const ACCEPT = ".pdf,.docx";

export default function AdminView({ docs, reload }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [lastDelete, setLastDelete] = useState(null);
  // Chosen files wait here until the user commits them, so there is a moment
  // in between to name each one. Uploading straight from the picker is what
  // made the title unsettable: by the time you saw the row, it was indexed.
  const [pending, setPending] = useState([]);
  const fileInput = useRef(null);

  function stage(files) {
    setError(null);
    setPending((current) => [
      ...current,
      ...files.map((file) => ({ file, title: "" })),
    ]);
  }

  function retitle(index, title) {
    setPending((current) =>
      current.map((item, i) => (i === index ? { ...item, title } : item))
    );
  }

  async function upload() {
    setError(null);
    for (const { file, title } of pending) {
      setBusy(`Indexing ${file.name}…`);
      try {
        await api.uploadDocument(file, title.trim());
      } catch (e) {
        setError(`${file.name}: ${e.message}`);
      }
    }
    setPending([]);
    setBusy(null);
    reload();
  }

  async function remove(doc) {
    setError(null);
    setBusy(`Removing ${doc.title}…`);
    try {
      setLastDelete({ ...(await api.deleteDocument(doc.id)), title: doc.title });
    } catch (e) {
      setError(e.message);
    }
    setBusy(null);
    reload();
  }

  return (
    <div>
      {/* WCAG 2.2 dragging-movements: drag is an enhancement, never the only
          path — the whole zone is also a button that opens the file picker. */}
      <button
        onClick={() => fileInput.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          stage([...e.dataTransfer.files]);
        }}
        className={`mb-5 flex w-full cursor-pointer flex-col items-center rounded-lg border-2 border-dashed p-10 transition-colors duration-150 ${
          over ? "border-accent bg-accent/5" : "border-line hover:border-edge"
        }`}
      >
        <Upload className={`h-8 w-8 ${over ? "text-accent" : "text-dim"}`} />
        <span className="mt-3 font-medium">
          Drop files here, or click to browse
        </span>
        <span className="mt-1 text-sm text-dim">
          PDF and Word documents, up to 50 MB
        </span>
      </button>
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPT}
        multiple
        className="sr-only"
        onChange={(e) => {
          stage([...e.target.files]);
          e.target.value = ""; // so re-picking the same file fires onChange
        }}
      />

      {/* Staged but not yet indexed: one title box per file, because a title
          belongs to a document and there may be several. */}
      {pending.length > 0 && (
        <div className="card mb-5 p-4">
          <div className="mb-3 text-xs font-medium uppercase tracking-wide text-dim">
            Ready to upload ({pending.length})
          </div>

          <ul className="space-y-3">
            {pending.map(({ file, title }, i) => (
              <li key={`${file.name}-${i}`} className="rounded-md bg-muted p-3">
                <div className="mb-2 flex items-center gap-3">
                  <Doc className="h-4 w-4 shrink-0 text-dim" />
                  <span className="flex-1 truncate font-mono text-xs text-dim">
                    {file.name}
                  </span>
                  <button
                    onClick={() =>
                      setPending((c) => c.filter((_, j) => j !== i))
                    }
                    aria-label={`Remove ${file.name}`}
                    className="cursor-pointer rounded p-1 text-dim transition-colors duration-150 hover:bg-danger/15 hover:text-danger"
                  >
                    <Trash className="h-3.5 w-3.5" />
                  </button>
                </div>
                <label htmlFor={`title-${i}`} className="sr-only">
                  Title for {file.name}
                </label>
                <input
                  id={`title-${i}`}
                  dir="auto"
                  value={title}
                  onChange={(e) => retitle(i, e.target.value)}
                  placeholder="Title — blank uses the document's own, then its filename"
                  className="w-full rounded-md border border-line bg-card px-3 py-2 text-sm placeholder:text-dim"
                />
              </li>
            ))}
          </ul>

          <div className="mt-4 flex items-center gap-2">
            <button
              onClick={upload}
              disabled={!!busy}
              className="cursor-pointer rounded-lg bg-accent px-5 py-2 font-medium text-accent-ink transition-colors duration-150 hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? "Uploading…" : `Upload ${pending.length}`}
            </button>
            <button
              onClick={() => setPending([])}
              disabled={!!busy}
              className="cursor-pointer rounded-lg border border-line px-4 py-2 text-sm text-dim transition-colors duration-150 hover:border-edge hover:text-fg disabled:opacity-40"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {busy && (
        <div className="card mb-5 flex items-center gap-2.5 p-3 text-sm" aria-live="polite">
          <Spinner className="h-4 w-4 text-accent" />
          {busy}
        </div>
      )}

      {error && (
        <div className="card mb-5 flex items-start gap-2.5 border-danger/50 p-4 text-sm">
          <Alert className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
          <span>{error}</span>
        </div>
      )}

      {/* The proof that both indexes were cleaned, not just the database. */}
      {lastDelete && (
        <div className="card mb-5 border-accent/40 p-4" aria-live="polite">
          <div className="flex items-center gap-2">
            <Check className="h-4 w-4 text-accent" />
            <span className="text-sm font-medium">
              Removed “{lastDelete.title}” from every index
            </span>
            <button
              onClick={() => setLastDelete(null)}
              className="ml-auto cursor-pointer text-xs text-dim hover:text-fg"
            >
              dismiss
            </button>
          </div>
          <div className="mt-3 grid grid-cols-3 gap-3 font-mono text-sm">
            {[
              ["chunks", lastDelete.chunks_removed, "SQLite"],
              ["postings", lastDelete.postings_removed, "inverted index"],
              ["vectors", lastDelete.vectors_removed, "Chroma"],
            ].map(([label, n, store]) => (
              <div key={label} className="rounded-md bg-muted p-3">
                <div className="text-lg text-accent">{n}</div>
                <div className="text-xs text-dim">{label}</div>
                <div className="mt-0.5 font-sans text-xs text-dim">{store}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {docs.length === 0 ? (
        <div className="card p-8 text-center">
          <Doc className="mx-auto h-8 w-8 text-dim" />
          <p className="mt-3 font-medium">No documents indexed</p>
          <p className="mt-1.5 text-sm text-dim">
            Upload a PDF or Word file above and it will be parsed, chunked, and
            written to both indexes.
          </p>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <caption className="sr-only">Indexed documents</caption>
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-dim">
                <th scope="col" className="px-4 py-3 font-medium">Title</th>
                <th scope="col" className="px-4 py-3 font-medium">Type</th>
                <th scope="col" className="px-4 py-3 text-right font-medium">Chunks</th>
                <th scope="col" className="px-4 py-3 text-right font-medium">Words</th>
                <th scope="col" className="px-4 py-3 font-medium">Added</th>
                <th scope="col" className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id} className="border-b border-line last:border-0">
                  <td className="px-4 py-3">
                    <div className="font-medium">{d.title}</div>
                    <div className="font-mono text-xs text-dim">{d.filename}</div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs uppercase text-dim">
                      {d.file_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono">{d.n_chunks}</td>
                  <td className="px-4 py-3 text-right font-mono">
                    {d.n_words.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-dim">
                    {new Date(d.added_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => remove(d)}
                      aria-label={`Delete ${d.title}`}
                      className="cursor-pointer rounded-md p-2 text-dim transition-colors duration-150 hover:bg-danger/15 hover:text-danger"
                    >
                      <Trash className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
