import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Header } from "../../components/Header";

const baseStats = {
  total_images: 100,
  images_with_faces: 80,
  total_faces: 200,
  num_clusters: 5,
  num_noise_faces: 10,
};

function renderHeader(props: Record<string, unknown> = {}) {
  return render(
    <Header
      stats={baseStats}
      canUndo={false}
      saving={false}
      mutating={false}
      saveResult={null}
      onUndo={vi.fn()}
      onOpenSave={vi.fn()}
      onOpenSettings={vi.fn()}
      {...props}
    />,
  );
}

describe("Header", () => {
  it("renders the title", () => {
    renderHeader();
    expect(screen.getByText("Visage Review")).toBeDefined();
  });

  it("renders stats", () => {
    renderHeader();
    expect(screen.getByText(/5 clusters/)).toBeDefined();
    expect(screen.getByText(/80 images/)).toBeDefined();
    expect(screen.getByText(/10 noise/)).toBeDefined();
  });

  it("disables undo button when canUndo is false", () => {
    renderHeader({ canUndo: false });
    expect(screen.getByText(/Undo/).closest("button")).toBeDisabled();
  });

  it("enables undo button when canUndo is true", () => {
    renderHeader({ canUndo: true });
    expect(screen.getByText(/Undo/).closest("button")).not.toBeDisabled();
  });

  it("calls onUndo when undo button is clicked", () => {
    const onUndo = vi.fn();
    renderHeader({ onUndo, canUndo: true });
    fireEvent.click(screen.getByText(/Undo/));
    expect(onUndo).toHaveBeenCalledTimes(1);
  });

  it("calls onOpenSave when save button is clicked", () => {
    const onOpenSave = vi.fn();
    renderHeader({ onOpenSave });
    fireEvent.click(screen.getByText("Save to Disk"));
    expect(onOpenSave).toHaveBeenCalledTimes(1);
  });

  it("shows saving state and disables save button", () => {
    renderHeader({ saving: true });
    expect(screen.getByText("Saving...")).toBeDefined();
    expect(screen.getByText("Saving...").closest("button")).toBeDisabled();
  });

  it("shows save result text when provided", () => {
    renderHeader({ saveResult: "42 files saved" });
    expect(screen.getByText("42 files saved")).toBeDefined();
  });

  it("disables undo while mutating even if canUndo is true", () => {
    renderHeader({ canUndo: true, mutating: true });
    expect(screen.getByText(/Undo/).closest("button")).toBeDisabled();
  });

  it("renders settings gear button", () => {
    const onOpenSettings = vi.fn();
    renderHeader({ onOpenSettings });
    const settingsBtn = screen.getByLabelText("Settings");
    expect(settingsBtn).toBeDefined();
    fireEvent.click(settingsBtn);
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });
});
