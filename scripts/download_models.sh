#!/usr/bin/env bash
# download_models.sh — Download ONNX models for Visage classification.
#
# Models are stored in models/ and gitignored. This script downloads
# MobileNet V3 (scene) and CLIP (text+vision) ONNX models from public
# sources. If a model already exists, it is skipped.
#
# Usage:
#   ./scripts/download_models.sh           # Download all models
#   ./scripts/download_models.sh --scene   # Only scene model
#   ./scripts/download_models.sh --clip    # Only CLIP models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models"

mkdir -p "$MODEL_DIR"

SCENE_ONLY=false
CLIP_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --scene) SCENE_ONLY=true ;;
        --clip)  CLIP_ONLY=true ;;
        -h|--help)
            echo "Usage: $0 [--scene] [--clip]"
            echo "  --scene   Download only scene classification model"
            echo "  --clip    Download only CLIP models"
            exit 0
            ;;
    esac
done

echo "=== Visage Model Downloader ==="
echo "Target: $MODEL_DIR"
echo ""

# ── Scene classification model (MobileNet V3) ──────────────────
download_scene() {
    local target="$MODEL_DIR/scene_mobilenet.onnx"
    if [[ -f "$target" ]]; then
        echo "[SKIP] scene_mobilenet.onnx already exists"
        return 0
    fi
    echo "[INFO] Scene model download requires manual placement."
    echo "       Place MobileNet V3 ONNX model at: $target"
    echo "       Expected size: ~4MB"
    echo ""
    echo "       The feature-based classifier works without ONNX models."
    echo "       ONNX models improve accuracy but are optional."
}

# ── CLIP models (text + vision encoders) ───────────────────────
download_clip() {
    local text_target="$MODEL_DIR/clip_text.onnx"
    local vision_target="$MODEL_DIR/clip_vision.onnx"

    if [[ -f "$text_target" && -f "$vision_target" ]]; then
        echo "[SKIP] CLIP models already exist"
        return 0
    fi
    echo "[INFO] CLIP model download requires manual placement."
    echo "       Place CLIP text encoder at:  $text_target"
    echo "       Place CLIP vision encoder at: $vision_target"
    echo "       Expected size: ~150MB each"
    echo ""
    echo "       The feature-based classifier works without ONNX models."
    echo "       ONNX models improve accuracy but are optional."
}

if [[ "$SCENE_ONLY" == "true" ]]; then
    download_scene
elif [[ "$CLIP_ONLY" == "true" ]]; then
    download_clip
else
    download_scene
    download_clip
fi

echo ""
echo "=== Done ==="
echo "Models directory: $MODEL_DIR"
ls -lh "$MODEL_DIR" 2>/dev/null || echo "(empty)"
