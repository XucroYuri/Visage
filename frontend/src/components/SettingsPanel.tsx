import { useState, useEffect, useCallback, useRef } from "react";
import { useSettingsStore } from "../store/settings";

// ── Types ──────────────────────────────────────────────────────

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  inputDir: string;
  embeddingBackend: string;
  totalImages: number;
  imagesWithFaces: number;
}

type TabId = "input" | "output";

// ── Component ──────────────────────────────────────────────────

export function SettingsPanel({
  open,
  onClose,
  inputDir,
  embeddingBackend,
  totalImages,
  imagesWithFaces,
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
    "fixed right-0 top-0 h-full w-96 bg-white shadow-xl z-50",
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
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-200 shrink-0">
          <h2 className="text-base font-semibold text-gray-900">Settings</h2>
          <button
            onClick={beginClose}
            className="p-1.5 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
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
          {(["input", "output"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 text-sm font-medium rounded-full transition-colors ${
                activeTab === tab
                  ? "bg-blue-600 text-white"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            >
              {tab === "input" ? "Input" : "Output"}
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
  return (
    <div className="space-y-5">
      <Field label="Input Directory">
        <p
          className="text-sm text-gray-600 font-mono truncate bg-gray-50 rounded px-2.5 py-1.5 border border-gray-200"
          title={inputDir}
        >
          {inputDir}
        </p>
      </Field>

      <Field label="Embedding Backend">
        <p className="text-sm text-gray-700">{embeddingBackend}</p>
      </Field>

      <Field label="Statistics">
        <div className="space-y-1.5">
          <StatRow label="Total images scanned" value={totalImages} />
          <StatRow label="Images with faces" value={imagesWithFaces} />
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

  return (
    <div className="space-y-5">
      {/* Output Directory */}
      <Field label="Output Directory">
        <input
          type="text"
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          placeholder="Default: input_dir/visage_output"
          className="w-full text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded px-2.5 py-1.5 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow"
        />
      </Field>

      {/* Folder Prefix */}
      <Field label="Folder Prefix">
        <input
          type="text"
          value={folderPrefix}
          onChange={(e) => setFolderPrefix(e.target.value)}
          className="w-full text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-shadow"
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
