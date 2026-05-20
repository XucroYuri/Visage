import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PhotoCard } from "../../components/PhotoCard";
import type { PhotoInfo, ClusterInfo } from "../../api";

const basePhoto: PhotoInfo = {
  path: "/photos/img001.jpg",
  width: 640,
  height: 480,
  faces: [
    { top: 10, right: 100, bottom: 120, left: 5 },
    { top: 50, right: 200, bottom: 180, left: 80 },
  ],
};

const otherClusters: ClusterInfo[] = [
  {
    id: 1,
    name: "Beach Trip",
    photos: [],
    photo_count: 5,
    thumbnail: null,
    confidence: 0.9,
  },
  {
    id: 2,
    name: "Family",
    photos: [],
    photo_count: 12,
    thumbnail: null,
    confidence: 0.85,
  },
];

describe("PhotoCard", () => {
  it("renders the photo image with correct alt text", () => {
    render(<PhotoCard photo={basePhoto} />);
    const img = screen.getByAltText("img001.jpg") as HTMLImageElement;
    expect(img).toBeDefined();
    expect(img.src).toContain("size=thumb");
  });

  it("renders the filename", () => {
    render(<PhotoCard photo={basePhoto} />);
    expect(screen.getByText("img001.jpg")).toBeDefined();
  });

  it("renders the image with encoded path in src", () => {
    render(<PhotoCard photo={basePhoto} />);
    const img = screen.getByAltText("img001.jpg") as HTMLImageElement;
    expect(img.src).toContain("path=");
  });

  it("shows Remove button when onRemove is provided", () => {
    render(<PhotoCard photo={basePhoto} onRemove={vi.fn()} />);
    expect(screen.getByText("Remove")).toBeDefined();
  });

  it("calls onRemove when remove button is clicked", () => {
    const onRemove = vi.fn();
    render(<PhotoCard photo={basePhoto} onRemove={onRemove} />);
    fireEvent.click(screen.getByText("Remove"));
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("shows Move button when onMove is provided", () => {
    render(
      <PhotoCard
        photo={basePhoto}
        onMove={vi.fn()}
        otherClusters={otherClusters}
        nextClusterId={99}
      />,
    );
    expect(screen.getByText("Move")).toBeDefined();
  });

  it("opens move dropdown on Move button click", () => {
    render(
      <PhotoCard
        photo={basePhoto}
        onMove={vi.fn()}
        otherClusters={otherClusters}
        nextClusterId={99}
      />,
    );
    fireEvent.click(screen.getByText("Move"));
    expect(screen.getByText(/Beach Trip/)).toBeDefined();
    expect(screen.getByText(/Family/)).toBeDefined();
  });

  it("calls onMove with cluster id when a cluster is clicked", () => {
    const onMove = vi.fn();
    render(
      <PhotoCard
        photo={basePhoto}
        onMove={onMove}
        otherClusters={otherClusters}
        nextClusterId={99}
      />,
    );
    fireEvent.click(screen.getByText("Move"));
    fireEvent.click(screen.getByText(/Beach Trip/));
    expect(onMove).toHaveBeenCalledWith(1);
  });

  it("shows + New cluster option in move dropdown", () => {
    render(
      <PhotoCard
        photo={basePhoto}
        onMove={vi.fn()}
        otherClusters={otherClusters}
        nextClusterId={99}
      />,
    );
    fireEvent.click(screen.getByText("Move"));
    expect(screen.getByText("+ New cluster")).toBeDefined();
  });

  it("shows selection checkbox in selection mode", () => {
    render(
      <PhotoCard
        photo={basePhoto}
        selectionMode
        selected={false}
        onSelectToggle={vi.fn()}
      />,
    );
    expect(screen.getByRole("checkbox")).toBeDefined();
  });

  it("checkbox reflects selected state", () => {
    render(
      <PhotoCard
        photo={basePhoto}
        selectionMode
        selected={true}
        onSelectToggle={vi.fn()}
      />,
    );
    expect(screen.getByRole("checkbox")).toBeChecked();
  });
});
