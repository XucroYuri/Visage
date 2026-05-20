import { useEffect, useState } from "react";
import type { ToastItem, ToastType } from "../store/toast";

interface ToastContainerProps {
  toasts: ToastItem[];
  onDismiss: (id: number) => void;
}

const COLORS: Record<ToastType, string> = {
  success: "bg-green-600",
  error: "bg-red-600",
  info: "bg-blue-600",
};

const ICONS: Record<ToastType, string> = {
  success: "\u2713",
  error: "\u2717",
  info: "\u2139",
};

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  return (
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: ToastItem;
  onDismiss: (id: number) => void;
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Entrance animation trigger
    const enterFrame = requestAnimationFrame(() => setVisible(true));

    // Auto-dismiss after 4s (3.5s visible + 0.5s exit)
    const timer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onDismiss(toast.id), 300);
    }, 3500);

    return () => {
      cancelAnimationFrame(enterFrame);
      clearTimeout(timer);
    };
  }, [toast.id, onDismiss]);

  return (
    <div
      role="alert"
      className={`
        pointer-events-auto flex items-center gap-2 px-4 py-2.5 rounded-lg shadow-lg
        text-white text-sm font-medium max-w-sm
        transition-all duration-300 ease-out
        ${COLORS[toast.type]}
        ${visible ? "translate-x-0 opacity-100" : "translate-x-8 opacity-0"}
      `}
    >
      <span className="text-base shrink-0">{ICONS[toast.type]}</span>
      <span className="truncate">{toast.text}</span>
      <button
        onClick={() => {
          setVisible(false);
          setTimeout(() => onDismiss(toast.id), 300);
        }}
        className="ml-auto shrink-0 text-white/70 hover:text-white text-sm leading-none"
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
}
