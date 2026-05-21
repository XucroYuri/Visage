import { useState, useEffect, useCallback, useRef } from "react";
import { useSettingsStore } from "../store/settings";
import { useReclusterMutation } from "../store/workspace";

// ── Types ──────────────────────────────────────────────────────

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  inputDir: string;
  embeddingBackend: string;
  totalImages: number;
  imagesWithFaces: number;
  clusterMethod: string;
  mergeThreshold: number;
}

type TabId = "input" | "output" | "clustering";

// ── Component ──────────────────────────────────────────────────

export function SettingsPanel({
  open,
  onClose,
  inputDir,
  embeddingBackend,
  totalImages,
  imagesWithFaces,
  clusterMethod,
  mergeThreshold,
}: SettingsPanelProps) {
  // ── Close animation state ──────────────────────────────────
  // When `open` becomes true, render the panel in its "entering" state,
  // then flip to "visible" on the next frame so CSS transitions fire.
  // When the user triggers close, flip to "leaving", wait for the
  // transition to finish, then call onClose() so the parent sets open=false.

  type AnimPhase = "entering" | "visible" | "leaving";

  const [phase, setPhase] = useState<AnimPhase>("entering");
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setPhase("entering");
      const raf = requestAnimationFrame(() => {
        setPhase("visible");
      });
      return () => cancelAnimationFrame(raf);
    }
  }, [open]);

  const beginClose = useCallback(() => {
    setPhase("leaving");
  }, []);

  const handleTransitionEnd = useCallback(() => {
    if (phase === "leaving") {
      onClose();
    }
  }, [phase, onClose]);

  // Close on Escape key
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") beginClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, beginClose]);

  // ── Tab state ──────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<TabId>("input");

  if (!open && phase !== "leaving") return null;

  // ── CSS transition classes ─────────────────────────────────
  const isVisible = phase === "visible";

  const backdropClass = [
    "fixed inset-0 bg-black/40 z-40 transition-opacity duration-300 ease-out",
    isVisible ? "opacity-100" : "opacity-0",
  ].join(" ");

  const panelClass = [
    "fixed right-0 top-0 h-full w-96 bg-white dark:bg-slate-800 shadow-xl z-50",
    "flex flex-col",
    "transition-transform duration-300 ease-out",
    isVisible ? "translate-x-0" : "translate-x-full",
  ].join(" ");

  return (
    <>
      {/* Backdrop */}
      <div
        className={backdropClass}
        onClick={beginClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        ref={panelRef}
        className={panelClass}
        onTransitionEnd={handleTransitionEnd}
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-200 dark:border-slate-700 dark:border-slate-700 shrink-0">
          <h2 className="text-base font-semibold text-gray-900 dark:text-slate-100">Settings</h2>
          <button
            onClick={beginClose}
            className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 dark:text-slate-500 dark:hover:text-slate-300 dark:hover:bg-slate-700 hover:bg-gray-100 transition-colors"
            aria-label="Close settings"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 px-5 pt-3.5 pb-3 border-b border-gray-100 shrink-0">
          {(["input", "output", "clustering"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 text-sm font-medium rounded-full transition-colors ${
                activeTab === tab
                  ? "bg-blue-600 text-white"
                  : "text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-700 hover:bg-gray-100"
              }`}
            >
              {tab === "input" ? "Input" : tab === "output" ? "Output" : "Clustering"}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {activeTab === "input" ? (
            <InputTab
              inputDir={inputDir}
              embeddingBackend={embeddingBackend}
              totalImages={totalImages}
              imagesWithFaces={imagesWithFaces}
            />
          ) : activeTab === "clustering" ? (
            <ClusteringTab
              clusterMethod={clusterMethod}
              mergeThreshold={mergeThreshold}
            />
          ) : (
            <OutputTab />
          )}
        </div>
      </div>
    </>
  );
}

// ── Input Tab ──────────────────────────────────────────────────

function InputTab({
  inputDir,
  embeddingBackend,
  totalImages,
  imagesWithFaces,
}: {
  inputDir: string;
  embeddingBackend: string;
  totalImages: number;
  imagesWithFaces: number;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopyPath = useCallback(() => {
    navigator.clipboard.writeText(inputDir).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [inputDir]);

  const detRate = totalImages > 0 ? Math.round((imagesWithFaces / totalImages) * 100) : 0;

  return (
    <div className="space-y-5">
      <Field label="Input Directory">
        <button
          onClick={handleCopyPath}
          className="w-full text-left text-sm text-gray-600 font-mono truncate bg-gray-50 rounded px-2.5 py-1.5 border border-gray-200 dark:border-slate-700 hover:border-blue-300 hover:bg-blue-50/30 transition-colors group relative"
          title="Click to copy path"
        >
          {inputDir}
          <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] font-sans font-medium text-gray-400 group-hover:text-gray-600 transition-colors">
            {copied ? (
              <span className="text-green-600">Copied!</span>
            ) : (
              "Copy"
            )}
          </span>
        </button>
      </Field>

      <Field label="Embedding Backend">
        <p className="text-sm text-gray-700">{embeddingBackend}</p>
      </Field>

      <Field label="Statistics">
        <div className="space-y-3">
          <StatRow label="Total images scanned" value={totalImages} />
          <StatRow label="Images with faces" value={imagesWithFaces} />
          <StatRow label="No face found" value={totalImages - imagesWithFaces} />
          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-gray-500">Face detection rate</span>
              <span className="font-semibold text-gray-800 tabular-nums">{detRate}%</span>
            </div>
            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${detRate}%`,
                  backgroundColor: detRate > 80 ? "#22c55e" : detRate > 50 ? "#eab308" : "#ef4444",
                }}
              />
            </div>
          </div>
        </div>
      </Field>
    </div>
  );
}

// ── Output Tab ─────────────────────────────────────────────────

function OutputTab() {
  const outputDir = useSettingsStore((s) => s.outputDir);
  const setOutputDir = useSettingsStore((s) => s.setOutputDir);
  const folderPrefix = useSettingsStore((s) => s.folderPrefix);
  const setFolderPrefix = useSettingsStore((s) => s.setFolderPrefix);
  const copyMode = useSettingsStore((s) => s.copyMode);
  const setCopyMode = useSettingsStore((s) => s.setCopyMode);
  const includeUnclustered = useSettingsStore((s) => s.includeUnclustered);
  const setIncludeUnclustered = useSettingsStore((s) => s.setIncludeUnclustered);
  const includeNoFaces = useSettingsStore((s) => s.includeNoFaces);
  const setIncludeNoFaces = useSettingsStore((s) => s.setIncludeNoFaces);

  const handleResetDefaults = useCallback(() => {
    setOutputDir("");
    setFolderPrefix("person_");
    setCopyMode(true);
    setIncludeUnclustered(false);
    setIncludeNoFaces(false);
  }, [setOutputDir, setFolderPrefix, setCopyMode, setIncludeUnclustered, setIncludeNoFaces]);

  const effectiveDir = outputDir || "input_dir/visage_output";

  return (
    <div className="space-y-5">
      {/* Output Directory */}
      <Field label="Output Directory">
        <div className="flex gap-2">
          <input
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            placeholder="Default: input_dir/visage_output"
            className="flex-1 min-w-0 text-sm text-gray-700 bg-gray-50 border border-gray-200 dark:border-slate-700 rounded px-2.5 py-1.5 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow"
          />
        </div>
      </Field>

      {/* Folder Prefix */}
      <Field label="Folder Prefix">
        <input
          type="text"
          value={folderPrefix}
          onChange={(e) => setFolderPrefix(e.target.value)}
          className="w-full text-sm text-gray-700 bg-gray-50 border border-gray-200 dark:border-slate-700 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow"
        />
      </Field>

      {/* Copy Mode */}
      <Field label="Copy Mode">
        <div className="flex gap-4">
          <RadioLabel
            checked={copyMode}
            onChange={() => setCopyMode(true)}
            label="Copy files (safe)"
            id="copy-safe"
          />
          <RadioLabel
            checked={!copyMode}
            onChange={() => setCopyMode(false)}
            label="Move files"
            id="copy-move"
          />
        </div>
      </Field>

      {/* Include options */}
      <div className="space-y-3">
        <Field>
          <CheckboxLabel
            checked={includeUnclustered}
            onChange={(e) => setIncludeUnclustered(e.target.checked)}
            label="Include _unclustered folder"
            id="include-unclustered"
          />
        </Field>

        <Field>
          <CheckboxLabel
            checked={includeNoFaces}
            onChange={(e) => setIncludeNoFaces(e.target.checked)}
            label="Include _no_faces folder"
            id="include-nofaces"
          />
        </Field>
      </div>

      {/* ── Folder preview ───────────────────────────────────── */}
      <Field label="Expected Folder Structure">
        <div className="bg-gray-50 border border-gray-200 dark:border-slate-700 rounded p-2.5 text-xs font-mono text-gray-600 leading-relaxed">
          <div className="text-gray-700">{effectiveDir}/</div>
          {includeUnclustered && <div className="pl-3 text-gray-500">_unclustered/</div>}
          {includeNoFaces && <div className="pl-3 text-gray-500">_no_faces/</div>}
          <div className="pl-3 text-gray-500">{folderPrefix || "person_"}1/</div>
          <div className="pl-6 text-gray-400">photo_001.jpg</div>
          <div className="pl-6 text-gray-400">&hellip;</div>
        </div>
      </Field>

      {/* ── Reset ─────────────────────────────────────────────── */}
      <div className="pt-1 border-t border-gray-100">
        <button
          type="button"
          onClick={handleResetDefaults}
          className="text-xs text-gray-400 hover:text-gray-600 hover:text-red-600 transition-colors"
        >
          Reset output settings to defaults
        </button>
      </div>
    </div>
  );
}

// ── Shared sub-components ──────────────────────────────────────

function Field({
  label,
  children,
}: {
  label?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      {label && (
        <label className="block text-xs font-medium text-gray-400 uppercase tracking-wide">
          {label}
        </label>
      )}
      {children}
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-gray-500">{label}</span>
      <span className="font-semibold text-gray-800 tabular-nums">
        {value.toLocaleString()}
      </span>
    </div>
  );
}

function RadioLabel({
  checked,
  onChange,
  label,
  id,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
  id: string;
}) {
  return (
    <label
      htmlFor={id}
      className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer select-none"
    >
      <input
        id={id}
        type="radio"
        name="copyMode"
        checked={checked}
        onChange={onChange}
        className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
      />
      {label}
    </label>
  );
}

// ── Clustering Tab ─────────────────────────────────────────────

interface ClusterParamProps {
  label: string;
  value: number | string;
  onChange: (v: number | string) => void;
  type?: "number" | "text";
  step?: number;
  min?: number;
  max?: number;
  options?: { value: string; label: string }[];
  help?: string;
}

function ClusterParam({
  label,
  value,
  onChange,
  type = "number",
  step,
  min,
  max,
  options,
  help,
}: ClusterParamProps) {
  return (
    <div className="space-y-1">
      <label className="text-xs font-medium text-gray-400 dark:text-slate-500 uppercase tracking-wide">
        {label}
      </label>
      {options ? (
        <select
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          className="w-full text-sm text-gray-700 bg-gray-50 border border-gray-200 dark:border-slate-700 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          value={value}
          onChange={(e) => {
            const v = type === "number" ? parseFloat(e.target.value) : e.target.value;
            onChange(v);
          }}
          step={step}
          min={min}
          max={max}
          className="w-full text-sm text-gray-700 bg-gray-50 border border-gray-200 dark:border-slate-700 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow"
        />
      )}
      {help && <p className="text-[11px] text-gray-400 leading-snug">{help}</p>}
    </div>
  );
}

function ClusteringTab({
  clusterMethod: initialClusterMethod,
  mergeThreshold: initialMergeThreshold,
}: {
  clusterMethod: string;
  mergeThreshold: number;
}) {
  const reclusterMutation = useReclusterMutation();

  const [method, setMethod] = useState(initialClusterMethod || "hdbscan");
  const [minSamples, setMinSamples] = useState(2);
  const [minClusterSize, setMinClusterSize] = useState(2);
  const [cse, setCse] = useState(0.0);
  const [csm, setCsm] = useState("eom");
  const [mergeThresh, setMergeThresh] = useState(initialMergeThreshold || 0.80);
  const [smallMergeThresh, setSmallMergeThresh] = useState(0.75);
  const [minReliableSize, setMinReliableSize] = useState(10);
  const [hfWeight, setHfWeight] = useState(0.0);

  const handleRecluster = useCallback(() => {
    const settings = {
      cluster_method: method,
      min_samples: minSamples,
      min_cluster_size: minClusterSize,
      cluster_selection_epsilon: cse,
      cluster_selection_method: csm,
      merge_threshold: mergeThresh,
      small_merge_threshold: smallMergeThresh,
      min_reliable_size: minReliableSize,
      head_feature_weight: hfWeight,
    };
    reclusterMutation.mutate(settings);
  }, [
    method, minSamples, minClusterSize, cse, csm,
    mergeThresh, smallMergeThresh, minReliableSize, hfWeight,
    reclusterMutation,
  ]);

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-400 leading-snug">
        Adjust clustering parameters and re-run to improve results
        without re-scanning or re-generating embeddings.
      </p>

      <div className="space-y-3">
        <ClusterParam
          label="Cluster Method"
          value={method}
          onChange={(v) => setMethod(String(v))}
          options={[
            { value: "hdbscan", label: "HDBSCAN (recommended)" },
            { value: "dbscan", label: "DBSCAN (legacy)" },
          ]}
        />

        <ClusterParam
          label="Min Samples"
          value={minSamples}
          onChange={(v) => setMinSamples(Number(v))}
          min={1}
          max={20}
          help="Lower = more clusters (finer), Higher = fewer clusters (coarser)"
        />

        {method === "hdbscan" && (
          <>
            <ClusterParam
              label="Min Cluster Size"
              value={minClusterSize}
              onChange={(v) => setMinClusterSize(Number(v))}
              min={2}
              max={50}
              help="Minimum faces needed to form a new cluster"
            />

            <ClusterParam
              label="Cluster Selection Epsilon"
              value={cse}
              onChange={(v) => setCse(Number(v))}
              step={0.05}
              min={0}
              max={1}
              help="Higher values merge nearby clusters. Try 0.10–0.25 for fewer clusters"
            />

            <ClusterParam
              label="Selection Method"
              value={csm}
              onChange={(v) => setCsm(String(v))}
              options={[
                { value: "eom", label: "EOM (Excess of Mass)" },
                { value: "leaf", label: "Leaf (more clusters)" },
              ]}
              help="EOM produces fewer, more stable clusters"
            />
          </>
        )}

        <ClusterParam
          label="Merge Threshold"
          value={mergeThresh}
          onChange={(v) => setMergeThresh(Number(v))}
          step={0.05}
          min={0}
          max={1}
          help="Cosine similarity threshold (higher = stricter). Lower (0.70–0.80) merges more aggressively"
        />

        <ClusterParam
          label="Small Merge Threshold"
          value={smallMergeThresh}
          onChange={(v) => setSmallMergeThresh(Number(v))}
          step={0.05}
          min={0}
          max={1}
          help="Relaxed threshold for small clusters"
        />

        <ClusterParam
          label="Min Reliable Size"
          value={minReliableSize}
          onChange={(v) => setMinReliableSize(Number(v))}
          min={1}
          max={100}
          help="Clusters below this use the relaxed threshold"
        />

        <ClusterParam
          label="Head Feature Weight"
          value={hfWeight}
          onChange={(v) => setHfWeight(Number(v))}
          step={0.1}
          min={0}
          max={1}
          help="0 = ignore head pose features (recommended for AI art)"
        />
      </div>

      <div className="pt-3 border-t border-gray-100">
        <button
          onClick={handleRecluster}
          disabled={reclusterMutation.isPending}
          className="w-full px-4 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors flex items-center justify-center gap-2"
        >
          {reclusterMutation.isPending ? (
            <>
              <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Re-clustering...
            </>
          ) : (
            "Re-cluster"
          )}
        </button>
      </div>
    </div>
  );
}

function CheckboxLabel({
  checked,
  onChange,
  label,
  id,
}: {
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  label: string;
  id: string;
}) {
  return (
    <label
      htmlFor={id}
      className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer select-none"
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
      />
      {label}
    </label>
  );
}
