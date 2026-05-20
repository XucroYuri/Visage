import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ToastContainer } from "../../components/Toast";
import { useToastStore } from "../../store/toast";
import type { ToastItem } from "../../store/toast";

function resetStore() {
  useToastStore.setState({ toasts: [] });
}

beforeEach(resetStore);

describe("ToastContainer", () => {
  const onDismiss = vi.fn();

  it("renders empty when there are no toasts", () => {
    const { container } = render(
      <ToastContainer toasts={[]} onDismiss={onDismiss} />,
    );
    expect(container.firstElementChild?.childElementCount).toBe(0);
  });

  it("renders toast text", () => {
    const toasts: ToastItem[] = [
      { id: 1, type: "success", text: "Operation completed" },
    ];
    render(<ToastContainer toasts={toasts} onDismiss={onDismiss} />);
    expect(screen.getByText("Operation completed")).toBeDefined();
  });

  it("renders success icon (✓)", () => {
    const toasts: ToastItem[] = [
      { id: 1, type: "success", text: "Success" },
    ];
    render(<ToastContainer toasts={toasts} onDismiss={onDismiss} />);
    expect(screen.getByText("\u2713")).toBeDefined();
  });

  it("renders error icon (✗)", () => {
    const toasts: ToastItem[] = [
      { id: 1, type: "error", text: "Error" },
    ];
    render(<ToastContainer toasts={toasts} onDismiss={onDismiss} />);
    expect(screen.getByText("\u2717")).toBeDefined();
  });

  it("renders info icon (ℹ)", () => {
    const toasts: ToastItem[] = [
      { id: 1, type: "info", text: "Info" },
    ];
    render(<ToastContainer toasts={toasts} onDismiss={onDismiss} />);
    expect(screen.getByText("\u2139")).toBeDefined();
  });

  it("renders close button", () => {
    const toasts: ToastItem[] = [
      { id: 1, type: "info", text: "Info" },
    ];
    render(<ToastContainer toasts={toasts} onDismiss={vi.fn()} />);
    expect(screen.getByLabelText("Dismiss")).toBeDefined();
    expect(screen.getByText("\u00d7")).toBeDefined();
  });
});
