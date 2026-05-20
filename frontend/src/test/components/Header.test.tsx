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

describe("Header", () => {
  it("renders the title", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={false}
        saving={false}
        mutating={false}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText("Visage Review")).toBeDefined();
  });

  it("renders stats", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={false}
        saving={false}
        mutating={false}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText(/5 clusters/)).toBeDefined();
    expect(screen.getByText(/80 images/)).toBeDefined();
    expect(screen.getByText(/10 noise/)).toBeDefined();
  });

  it("disables undo button when canUndo is false", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={false}
        saving={false}
        mutating={false}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText(/Undo/).closest("button")).toBeDisabled();
  });

  it("enables undo button when canUndo is true", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={true}
        saving={false}
        mutating={false}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText(/Undo/).closest("button")).not.toBeDisabled();
  });

  it("calls onUndo when undo button is clicked", () => {
    const onUndo = vi.fn();
    render(
      <Header
        stats={baseStats}
        canUndo={true}
        saving={false}
        mutating={false}
        saveResult={null}
        onUndo={onUndo}
        onSave={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText(/Undo/));
    expect(onUndo).toHaveBeenCalledTimes(1);
  });

  it("calls onSave when save button is clicked", () => {
    const onSave = vi.fn();
    render(
      <Header
        stats={baseStats}
        canUndo={false}
        saving={false}
        mutating={false}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={onSave}
      />,
    );
    fireEvent.click(screen.getByText("Save to Disk"));
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it("shows saving state and disables save button", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={false}
        saving={true}
        mutating={false}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText("Saving...")).toBeDefined();
    expect(screen.getByText("Saving...").closest("button")).toBeDisabled();
  });

  it("shows save result text when provided", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={false}
        saving={false}
        mutating={false}
        saveResult="42 files saved"
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText("42 files saved")).toBeDefined();
  });

  it("disables undo while mutating even if canUndo is true", () => {
    render(
      <Header
        stats={baseStats}
        canUndo={true}
        saving={false}
        mutating={true}
        saveResult={null}
        onUndo={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByText(/Undo/).closest("button")).toBeDisabled();
  });
});
