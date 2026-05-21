"""Build the Visage engine binary using PyInstaller.

Usage:
    bash scripts/build-engine.sh

Output:
    src-tauri/binaries/visage-engine-{target-triple}
"""

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detect platform triple
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
  darwin)
    case "$ARCH" in
      arm64) TRIPLE="aarch64-apple-darwin" ;;
      x86_64) TRIPLE="x86_64-apple-darwin" ;;
      *) echo "Unknown arch: $ARCH"; exit 1 ;;
    esac
    ;;
  linux)
    case "$ARCH" in
      aarch64) TRIPLE="aarch64-unknown-linux-gnu" ;;
      x86_64) TRIPLE="x86_64-unknown-linux-gnu" ;;
      *) echo "Unknown arch: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

OUTPUT_DIR="$PROJECT_DIR/src-tauri/binaries"
OUTPUT_NAME="visage-engine-$TRIPLE"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_NAME"

echo "Building Visage engine for $TRIPLE..."
echo "Output: $OUTPUT_PATH"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

pip install pyinstaller

pyinstaller \
  --onefile \
  --name "$OUTPUT_NAME" \
  --distpath "$OUTPUT_DIR" \
  --workpath "$PROJECT_DIR/build/pyinstaller" \
  --specpath "$PROJECT_DIR/build/pyinstaller" \
  --add-data "src/visage:visage" \
  --hidden-import visage \
  --hidden-import visage.server \
  --hidden-import visage.server.app \
  --hidden-import visage.server.routes \
  --hidden-import visage.server.workspace \
  --hidden-import visage.detectors \
  --hidden-import visage.detectors.vision \
  --hidden-import visage.detectors.nms \
  --hidden-import sklearn \
  --hidden-import sklearn.cluster \
  --hidden-import sklearn.neighbors \
  --hidden-import numpy \
  --hidden-import PIL \
  --hidden-import PIL._imaging \
  --exclude matplotlib \
  --exclude tkinter \
  --exclude PyQt5 \
  --exclude PyQt6 \
  --exclude PySide2 \
  --exclude PySide6 \
  --exclude IPython \
  --exclude pandas \
  --exclude scipy.spatial \
  src/visage/serve.py

# PyInstaller adds the binary directly to distpath
# Rename to match Tauri's expected format (strip extension if any)
if [ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]; then
  echo "Binary built successfully: $OUTPUT_PATH"
  ls -lh "$OUTPUT_PATH"
elif [ -f "$OUTPUT_DIR/$OUTPUT_NAME.exe" ]; then
  mv "$OUTPUT_DIR/$OUTPUT_NAME.exe" "$OUTPUT_PATH"
  echo "Binary built successfully: $OUTPUT_PATH"
  ls -lh "$OUTPUT_PATH"
else
  echo "ERROR: Binary not found at $OUTPUT_DIR/$OUTPUT_NAME"
  ls "$OUTPUT_DIR/"
  exit 1
fi

# Clean up build artifacts
rm -rf "$PROJECT_DIR/build/pyinstaller"
rm -f "$PROJECT_DIR/$OUTPUT_NAME".spec

echo "Done."
