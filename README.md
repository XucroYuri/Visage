# Visage

macOS-native face clustering and photo sorting CLI. Scan a folder of photos, detect faces, group them by person, and organize into per-person subfolders.

## Features

- Hardware-accelerated face detection via macOS Vision framework
- 128-dimensional face identity embeddings via face_recognition (dlib)
- DBSCAN clustering with automatic eps estimation (`--auto-eps`)
- Per-cluster confidence scores (cosine similarity to centroid)
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
[3] Embed -- face_recognition generates 128-dim identity vectors
    |         (cached in SQLite; unchanged images skip re-computation)
    |
    v
[4] Cluster -- DBSCAN groups faces by person identity
    |
    v
[5] Organize -- copy/move photos into person_00/, person_01/, ...
```

Phase details:

1. **Scan** -- Walks the input directory for supported image files (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tif`, `.tiff`). Skips hidden directories, `visage_output`, and `.visage_cache`.

2. **Detect** -- Detects faces in each image using the macOS Vision framework (`VNDetectFaceRectanglesRequest`). Runs in parallel with configurable worker count. Filters by confidence threshold and minimum face size.

3. **Embed** -- Generates a 128-dimensional identity embedding for each detected face using face_recognition (dlib). Results are cached in a SQLite database keyed by file path and mtime fingerprint, so unchanged images skip re-computation on subsequent runs.

4. **Cluster** -- L2-normalizes all embeddings, then clusters them with DBSCAN. Supports a fixed epsilon threshold or automatic estimation via the k-distance elbow method. Faces that do not fit any cluster are labeled as noise.

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
pip install -e ".[dev]"
```

This installs the `visage` command-line tool and all dependencies, including pyobjc frameworks for Vision access and face_recognition for embeddings.

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
| `--model` | `small` | Face embedding model: `small` (fast) or `large` (accurate) |
| `--num-jitters` | `1` | Number of re-samples for embedding generation |

### Clustering

| Option | Default | Description |
|--------|---------|-------------|
| `--eps` | `0.5` | DBSCAN epsilon (max distance between embeddings in a cluster) |
| `--min-samples` | `2` | DBSCAN minimum samples per cluster |
| `--auto-eps` | off | Automatically estimate eps using k-distance elbow method |

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

## Architecture

Visage combines three core technologies:

- **Face detection**: macOS Vision framework via pyobjc. Runs `VNDetectFaceRectanglesRequest` with multi-threaded batch processing. Returns normalized bounding boxes converted to pixel coordinates.

- **Face embeddings**: face_recognition library (built on dlib). Produces a 128-dimensional vector for each detected face. Two model sizes are available: `small` (fast) and `large` (more accurate). Embeddings are cached in a SQLite database keyed by file path and mtime fingerprint.

- **Clustering**: scikit-learn DBSCAN on L2-normalized embeddings. After normalization, euclidean distance is monotonically related to cosine distance. Automatic eps estimation uses the k-distance elbow method (maximum perpendicular distance from the first-to-last line). Each cluster receives a confidence score computed as the mean cosine similarity of member embeddings to the cluster centroid.

## License

MIT
