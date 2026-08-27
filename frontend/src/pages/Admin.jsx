import { useRef, useState } from "react";
import * as api from "../api";
import { Alert, Check, Doc, Spinner, Trash, Upload } from "../components/icons";

const ACCEPT = ".pdf,.docx";

export default function AdminView({ docs, reload }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [lastDelete, setLastDelete] = useState(null);
  const fileInput = useRef(null);

  async function upload(files) {
    setError(null);
    for (const file of files) {
      setBusy(`Indexing ${file.name}…`);
      try {
        await api.uploadDocument(file);
      } catch (e) {
        setError(`${file.name}: ${e.message}`);
      }
    }
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
          upload([...e.dataTransfer.files]);
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
          upload([...e.target.files]);
          e.target.value = "";
        }}
      />

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
