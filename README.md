# Visage

macOS-native face clustering and photo sorting CLI. Scan a folder of photos, detect faces, group them by person, and organize into per-person subfolders. Includes an interactive web UI for manual review and correction.

## Features

- Hardware-accelerated face detection via macOS Vision framework
- Pluggable embedding backends: dlib (128-dim) and InsightFace/ArcFace (512-dim, optional)
- DBSCAN and HDBSCAN clustering with automatic eps estimation (`--auto-eps`)
- Face quality assessment -- Laplacian blur detection + size ratio filtering
- Per-cluster confidence scores (cosine similarity to centroid)
- **Interactive web UI** -- visual review, merge/split/rename clusters, face overlay
- SQLite embedding cache -- incremental processing on re-runs
- Checkpoint/resume for interrupted runs
- Rich terminal progress bars with plain-text fallback
- HEIC/HEIF image support (pillow-heif with macOS sips fallback)
- Copy-by-default (non-destructive), with `--move` option
- Dry-run mode to preview results before modifying files
- JSON output mode for scripting and automation
- Multi-face photos appear in every matching person folder

## How It Works

Visage runs a 5-phase pipeline:

```
Input folder
    |
    v
[1] Scan -- find all supported images recursively
    |
    v
[2] Detect -- macOS Vision finds faces (fast, hardware-accelerated)
    |
    v
[3] Embed -- pluggable backend generates identity vectors
    |         (dlib 128-dim or InsightFace 512-dim; cached in SQLite)
    |
    v
[4] Cluster -- DBSCAN or HDBSCAN groups faces by person identity
    |
    v
[5] Organize -- copy/move photos into person_00/, person_01/, ...
```

Phase details:

1. **Scan** -- Walks the input directory for supported image files (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tif`, `.tiff`). Skips hidden directories, `visage_output`, and `.visage_cache`.

2. **Detect** -- Detects faces in each image using the macOS Vision framework (`VNDetectFaceRectanglesRequest`). Runs in parallel with configurable worker count. Filters by confidence threshold and minimum face size.

3. **Embed** -- Generates identity embeddings for each detected face using a pluggable backend. The default dlib backend produces 128-dimensional vectors; the optional InsightFace/ArcFace backend produces 512-dimensional vectors for higher accuracy. Results are cached in a SQLite database keyed by file path and mtime fingerprint, so unchanged images skip re-computation on subsequent runs.

4. **Cluster** -- L2-normalizes all embeddings, then clusters them with DBSCAN or HDBSCAN. DBSCAN supports a fixed epsilon threshold or automatic estimation via the k-distance elbow method. HDBSCAN requires no eps parameter and handles clusters of varying density. Faces that do not fit any cluster are labeled as noise.

5. **Organize** -- Copies (or moves) photos into per-person subfolders under the output directory. A photo containing multiple people appears in each matching folder. Optionally includes `_unclustered` and `_no_faces` folders.

## Requirements

- macOS 13+ (Ventura or later)
- Python 3.10+
- cmake (required to build dlib, which face_recognition depends on)

Install cmake via Homebrew:

```bash
brew install cmake
```

## Installation

```bash
git clone https://github.com/user/Visage.git
cd Visage
uv sync --extra dev --extra insightface --extra web
```

This installs the `visage` command-line tool with all optional dependencies. Use `pip install -e ".[dev,insightface,web]"` if you prefer pip.

## Quick Start

Sort a folder of photos by person:

```bash
visage ~/Photos/Vacation
```

Results are placed in `~/Photos/Vacation/visage_output/` with subfolders `person_00/`, `person_01/`, and so on.

Preview results without modifying any files:

```bash
visage ~/Photos/Vacation --dry-run
```

Get machine-readable output:

```bash
visage ~/Photos/Vacation --json
```

Move files instead of copying:

```bash
visage ~/Photos/Vacation --move
```

Let Visage estimate the best clustering threshold automatically:

```bash
visage ~/Photos/Vacation --auto-eps
```

## Web UI (Review Mode)

Launch an interactive web interface to visually review and correct clustering results:

```bash
visage ~/Photos/Vacation --serve --backend insightface
```

This opens a browser at `http://localhost:8787` with:

- **Cluster sidebar** -- browse all detected person clusters with thumbnails and confidence scores
- **Face overlay** -- green bounding boxes show which face was detected in each photo
- **Merge** -- combine clusters that belong to the same person (merge mode or drag)
- **Move** -- reassign individual photos between clusters via dropdown menu
- **Remove** -- remove misidentified photos from a cluster (becomes unclustered)
- **Rename** -- click any cluster name to assign a person name
- **Undo** -- all operations are reversible with undo stack
- **Unclustered panel** -- review noise faces and assign them to clusters
- **Save to Disk** -- write the final organized folder structure

Pipeline progress is streamed live via Server-Sent Events with a 5-phase progress bar.

Web UI dependencies (`fastapi`, `uvicorn`) are installed with the `[web]` extra.

## CLI Reference

## CLI Reference

```
visage INPUT [OPTIONS]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Input folder containing photos |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output-dir` | `<input>/visage_output` | Output directory for sorted photos |
| `--move` | copy | Move files instead of copying |
| `--dry-run` | off | Show organization plan without modifying files |

### Detection

| Option | Default | Description |
|--------|---------|-------------|
| `--min-confidence` | `0.5` | Minimum face detection confidence (0--1) |
| `--max-workers` | `4` | Max parallel detection workers |

### Embedding

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `dlib` | Embedding backend: `dlib` (128-dim) or `insightface` (512-dim) |
| `--model` | `small` | Face embedding model: `small` (fast) or `large` (accurate, dlib only) |
| `--num-jitters` | `1` | Number of re-samples for embedding generation (dlib only) |

### Quality

| Option | Default | Description |
|--------|---------|-------------|
| `--min-quality` | `0` | Minimum face quality score 0--1 (0 = no filtering) |

### Clustering

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster-method` | `dbscan` | Clustering algorithm: `dbscan` or `hdbscan` |
| `--eps` | `0.5` | DBSCAN epsilon (max distance between embeddings in a cluster, DBSCAN only) |
| `--min-samples` | `2` | Minimum samples per cluster |
| `--auto-eps` | off | Automatically estimate eps using k-distance elbow method (DBSCAN only) |

### Include

| Option | Default | Description |
|--------|---------|-------------|
| `--include-unclustered` | off | Include `_unclustered/` folder for unmatched faces |
| `--include-no-faces` | off | Include `_no_faces/` folder for images without faces |

### Display

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | off | Output results as JSON |
| `-q`, `--quiet` | off | Suppress progress output |
| `-v`, `--verbose` | off | Show detailed log output (warnings and debug info) |
| `--config` | none | Path to TOML config file |
| `--version` | | Print version and exit |

### Serve (Web UI)

| Option | Default | Description |
|--------|---------|-------------|
| `--serve` | off | Start web review UI instead of batch processing |
| `--port` | `8787` | Port for the review web server |
| `--no-open` | off | Don't auto-open browser |

## Configuration

Visage reads settings from a TOML config file. Place a `visage.toml` in your input directory, or pass one explicitly with `--config path/to/config.toml`.

Priority order: CLI flags > `--config` file > `visage.toml` in input directory > defaults.

Example `visage.toml`:

```toml
[detection]
confidence = 0.6
min_face_size = 50

[embedding]
model = "large"
num_jitters = 2

[clustering]
eps = 0.45
min_samples = 3

[output]
copy_mode = true
folder_prefix = "person_"
include_unclustered = false
include_no_faces = false
```

## Tuning Tips

- **Too many clusters** (same person split): increase `--eps` (e.g., 0.6)
- **Too few clusters** (different people merged): decrease `--eps` (e.g., 0.3)
- **Missing faces**: lower `--min-confidence` (e.g., 0.3)
- **Better accuracy**: use `--model large --num-jitters 10` (slower)
- **Unsure about eps**: use `--auto-eps` to estimate automatically
- **Varying cluster densities**: use `--cluster-method hdbscan` (no eps parameter needed)
- **Blurry or small faces**: use `--min-quality 0.3` to filter low-quality face detections
- **Higher-accuracy embeddings**: use `--backend insightface` for 512-dim ArcFace embeddings

## Architecture

Visage combines four core technologies:

- **Face detection**: macOS Vision framework via pyobjc. Runs `VNDetectFaceRectanglesRequest` with multi-threaded batch processing. Returns normalized bounding boxes converted to pixel coordinates.

- **Face embeddings**: pluggable backend system (`EmbeddingBackend` protocol). The default dlib backend (via face_recognition) produces 128-dimensional vectors with `small` (fast) and `large` (more accurate) model sizes. The optional InsightFace/ArcFace backend produces 512-dimensional vectors for higher accuracy. Embeddings are cached in a SQLite database keyed by file path and mtime fingerprint.

- **Face quality**: Laplacian blur detection combined with face size ratio filtering. Each detected face receives a quality score from 0 to 1, usable with `--min-quality` to filter out blurry or small faces before clustering.

- **Clustering**: DBSCAN or HDBSCAN on L2-normalized embeddings. DBSCAN (scikit-learn) supports a fixed epsilon threshold or automatic estimation via the k-distance elbow method (maximum perpendicular distance from the first-to-last line). HDBSCAN handles clusters of varying density without requiring an eps parameter. After normalization, euclidean distance is monotonically related to cosine distance. Each cluster receives a confidence score computed as the mean cosine similarity of member embeddings to the cluster centroid.

## License

MIT
