const ENGINES = [
  { id: "vsm", label: "VSM", hint: "tf-idf cosine, lnc.ltc" },
  { id: "bm25", label: "BM25", hint: "probabilistic, k1/b/k3" },
  { id: "rag", label: "RAG", hint: "dense vectors + LLM" },
];

const MODES = [
  { id: "champion", label: "Champion lists", hint: "r highest-tf chunks per term" },
  {
    id: "elimination",
    label: "Index elimination",
    hint: "high-idf query terms only — scores omit the dropped terms",
  },
  { id: "exact", label: "Exact (safe)", hint: "every chunk containing a query term" },
];

function Segmented({ label, options, value, onChange }) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium uppercase tracking-wide text-dim">
        {label}
      </div>
      <div
        role="radiogroup"
        aria-label={label}
        className="inline-flex gap-1 rounded-lg border border-line bg-card p-1"
      >
        {options.map((o) => (
          <button
            key={o.id}
            role="radio"
            aria-checked={value === o.id}
            title={o.hint}
            onClick={() => onChange(o.id)}
            className={`seg ${value === o.id ? "seg-on" : "seg-off"}`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function SearchControls({ opts, set, models }) {
  const lexical = opts.engine !== "rag";

  return (
    <div className="flex flex-wrap items-start gap-6">
      <Segmented
        label="Engine"
        options={ENGINES}
        value={opts.engine}
        onChange={(engine) =>
          // PRF is rejected by the API for anything but VSM, so the control
          // must not be able to construct that request in the first place.
          set({ ...opts, engine, prf: engine === "vsm" ? opts.prf : false })
        }
      />

      {lexical && (
        <Segmented
          label="Top-K strategy"
          options={MODES}
          value={opts.mode}
          onChange={(mode) => set({ ...opts, mode })}
        />
      )}

      {opts.engine === "vsm" && (
        <div>
          <div className="mb-1.5 text-xs font-medium uppercase tracking-wide text-dim">
            Query expansion
          </div>
          <label className="inline-flex cursor-pointer select-none items-center gap-2.5 rounded-lg border border-line bg-card px-3 py-2">
            <input
              type="checkbox"
              checked={opts.prf}
              onChange={(e) => set({ ...opts, prf: e.target.checked })}
              className="h-4 w-4 cursor-pointer accent-accent"
            />
            <span className="text-sm">Rocchio PRF</span>
          </label>
        </div>
      )}

      {opts.engine === "rag" && (
        <div>
          <label
            htmlFor="model"
            className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-dim"
          >
            Model
          </label>
          {/* ponytail: native <datalist> — a searchable combobox over 386 ids
              with zero dependencies. A <select> that long is unusable. */}
          <input
            id="model"
            list="model-options"
            value={opts.model}
            placeholder="default"
            onChange={(e) => set({ ...opts, model: e.target.value })}
            className="w-56 rounded-lg border border-line bg-card px-3 py-2 font-mono text-sm placeholder:text-dim"
          />
          <datalist id="model-options">
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </div>
      )}
    </div>
  );
}
