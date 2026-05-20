import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ClusterRow } from "../../components/ClusterRow";
import type { ClusterInfo } from "../../api";

const baseCluster: ClusterInfo = {
  id: 3,
  name: "Vacation 2024",
  photos: [],
  photo_count: 15,
  thumbnail: null,
  confidence: 0.92,
};

describe("ClusterRow", () => {
  it("renders cluster name and info", () => {
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={false}
        checkedForMerge={false}
        editing={false}
        editValue=""
        onSelect={vi.fn()}
        onStartEdit={vi.fn()}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    expect(screen.getByText("Vacation 2024")).toBeDefined();
    expect(screen.getByText(/15 photos/)).toBeDefined();
    expect(screen.getByText(/conf: 0.92/)).toBeDefined();
  });

  it("calls onSelect when clicked", () => {
    const onSelect = vi.fn();
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={false}
        checkedForMerge={false}
        editing={false}
        editValue=""
        onSelect={onSelect}
        onStartEdit={vi.fn()}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Vacation 2024").closest("div[role='option']")!);
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("shows active styling when selected", () => {
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={true}
        mergeMode={false}
        checkedForMerge={false}
        editing={false}
        editValue=""
        onSelect={vi.fn()}
        onStartEdit={vi.fn()}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    const option = screen.getByRole("option");
    expect(option).toHaveAttribute("aria-selected", "true");
    expect(option.className).toContain("bg-blue-50");
  });

  it("shows checkbox in merge mode", () => {
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={true}
        checkedForMerge={false}
        editing={false}
        editValue=""
        onSelect={vi.fn()}
        onStartEdit={vi.fn()}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    expect(screen.getByRole("checkbox")).toBeDefined();
  });

  it("shows checkbox checked when selected for merge", () => {
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={true}
        checkedForMerge={true}
        editing={false}
        editValue=""
        onSelect={vi.fn()}
        onStartEdit={vi.fn()}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  it("shows edit input when editing", () => {
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={false}
        checkedForMerge={false}
        editing={true}
        editValue="New Name"
        onSelect={vi.fn()}
        onStartEdit={vi.fn()}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    const input = screen.getByDisplayValue("New Name") as HTMLInputElement;
    expect(input).toBeDefined();
  });

  it("calls onStartEdit when name is clicked", () => {
    const onStartEdit = vi.fn();
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={false}
        checkedForMerge={false}
        editing={false}
        editValue=""
        onSelect={vi.fn()}
        onStartEdit={onStartEdit}
        onEditChange={vi.fn()}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Vacation 2024"));
    expect(onStartEdit).toHaveBeenCalledTimes(1);
  });

  it("calls onEditChange when input changes", () => {
    const onEditChange = vi.fn();
    render(
      <ClusterRow
        cluster={baseCluster}
        selected={false}
        mergeMode={false}
        checkedForMerge={false}
        editing={true}
        editValue="Old"
        onSelect={vi.fn()}
        onStartEdit={vi.fn()}
        onEditChange={onEditChange}
        onSaveEdit={vi.fn()}
        onCancelEdit={vi.fn()}
      />,
    );
    fireEvent.change(screen.getByDisplayValue("Old"), {
      target: { value: "Updated" },
    });
    expect(onEditChange).toHaveBeenCalledWith("Updated");
  });
});
