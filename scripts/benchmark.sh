#!/usr/bin/env bash
# benchmark.sh — Run Visage pipeline performance benchmarks.
#
# Usage:
#   ./scripts/benchmark.sh [INPUT_DIR] [--count N]
#
# If INPUT_DIR is not provided, generates synthetic test images.
# Outputs timing and quality metrics to stdout and BENCHMARKS.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARKS_FILE="$PROJECT_ROOT/BENCHMARKS.md"

# ── Defaults ──────────────────────────────────────────────────────
INPUT_DIR=""
COUNT=100
BACKEND="auto"

# ── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)
            COUNT="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [INPUT_DIR] [--count N] [--backend auto|vision|scrfd|yunet]"
            exit 0
            ;;
        *)
            INPUT_DIR="$1"
            shift
            ;;
    esac
done

# ── Generate synthetic data if no input dir ───────────────────────
if [[ -z "$INPUT_DIR" ]]; then
    INPUT_DIR="$(mktemp -d)/benchmark_images"
    mkdir -p "$INPUT_DIR"
    echo "Generating $COUNT synthetic test images in $INPUT_DIR ..."
    python3 - "$INPUT_DIR" "$COUNT" <<'PYEOF'
import sys, os
from PIL import Image, ImageDraw
import random

output_dir, count = sys.argv[1], int(sys.argv[2])
random.seed(42)

for i in range(count):
    img = Image.new("RGB", (640, 480), color=(
        random.randint(100, 200),
        random.randint(100, 200),
        random.randint(100, 200),
    ))
    draw = ImageDraw.Draw(img)
    # Draw a simple face-like oval
    cx, cy = 320 + random.randint(-50, 50), 240 + random.randint(-50, 50)
    draw.ellipse([cx-60, cy-80, cx+60, cy+80], fill=(220, 190, 160), outline=(180, 150, 120))
    draw.ellipse([cx-30, cy-40, cx-10, cy-20], fill=(50, 50, 50))  # left eye
    draw.ellipse([cx+10, cy-40, cx+30, cy-20], fill=(50, 50, 50))  # right eye
    img.save(os.path.join(output_dir, f"bench_{i:04d}.jpg"), "JPEG")
PYEOF
fi

echo ""
echo "=== Visage Benchmark ==="
echo "Input:    $INPUT_DIR ($(find "$INPUT_DIR" -type f | wc -l | tr -d ' ') files)"
echo "Backend:  $BACKEND"
echo "Date:     $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# ── Run pipeline ──────────────────────────────────────────────────
cd "$PROJECT_ROOT"

START_TIME=$(python3 -c "import time; print(time.time())")

uv run python -m visage "$INPUT_DIR" --backend "$BACKEND" 2>&1 | tee /tmp/visage_bench.log

END_TIME=$(python3 -c "import time; print(time.time())")

ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
FILE_COUNT=$(find "$INPUT_DIR" -type f | wc -l | tr -d ' ')
PER_IMAGE=$(python3 -c "print(f'{$ELAPSED / max(int('$FILE_COUNT'), 1):.4f}')")

echo ""
echo "=== Results ==="
echo "Total time:    ${ELAPSED}s"
echo "Files:         $FILE_COUNT"
echo "Per image:     ${PER_IMAGE}s"

# ── Append to BENCHMARKS.md ───────────────────────────────────────
echo "" >> "$BENCHMARKS_FILE"
echo "## $(date -u +"%Y-%m-%d") — ${FILE_COUNT} images, backend=${BACKEND}" >> "$BENCHMARKS_FILE"
echo "- Total: ${ELAPSED}s" >> "$BENCHMARKS_FILE"
echo "- Per image: ${PER_IMAGE}s" >> "$BENCHMARKS_FILE"
echo "- Platform: $(uname -s) $(uname -m)" >> "$BENCHMARKS_FILE"

echo ""
echo "Results appended to $BENCHMARKS_FILE"
