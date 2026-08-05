# Visage -- AI Agent Onboarding Instructions

## Project Identity
- **Name**: Visage
- **Version**: 0.1.0
- **Type**: Original project by XucroYuri
- **Package**: `visage` (Python), CLI commands: `visage`, `visage-engine`
- **Stack**: Python 3.10+ (setuptools), React 19 + TypeScript (Vite), Rust (Tauri 2)
- **Remote**: `origin` = https://github.com/XucroYuri/Visage.git
- **Description**: macOS-native face clustering and photo sorter. Scans photos, detects faces via macOS Vision framework or dlib/InsightFace, groups by person using HDBSCAN/DBSCAN clustering, and organizes into folders. Supports batch mode and interactive Web UI with Tauri desktop wrapper.

## Quick Reference

```bash
uv sync --extra dev --extra insightface --extra web   # Install all deps
uv run pytest                                           # Run tests
uv run ruff check .                                     # Lint
uv run visage ~/Photos/ --web                           # Run with web UI
uv run visage ~/Photos/ --output-dir ./output           # Batch mode
```

## Onboarding Workflow (for new AI agents)

### Phase 1: Deep Exploration
1. Read `pyproject.toml`, `README.md`, `docs/architecture.md`
2. Check `git remote -v` and `git log -1 --format=%ci`
3. List top-level directory: `ls -la`
4. One-sentence summary: "A macOS-native face clustering CLI tool with Python Vision framework backend, HDBSCAN clustering, React web UI, and Tauri desktop wrapper"

### Phase 2: AGENTS.md / CLAUDE.md
- AGENTS.md and CLAUDE.md should already exist at the repo root
- If stale, regenerate them

### Phase 3: Security Configuration
- `.claude/settings.json` should already exist with `acceptEdits` + Bash whitelist
- Verify: `defaultMode: "acceptEdits"`, deny rules cover rm -rf/sudo/chown

### Phase 4: Build Verification
```bash
python3 --version        # Must be >= 3.10
uv sync --extra dev --extra insightface --extra web
uv run pytest
uv run ruff check .
```

### Phase 5: Cleanup
- Check for stale __pycache__ or .pyc files
- Check for empty directories or stale log files

### Phase 6: Push to GitHub
- Use `gh auth status` to verify authentication
- Push via: `git -c credential.helper='!gh auth git-credential' push origin main`

## Key Architecture

```
CLI (visage <input_dir> --output-dir ...)
  -> visage/cli.py (argparse)
    -> visage/config.py (VisageConfig dataclass, build_config, hwdetect)
      -> visage/pipeline.py (run_pipeline: scan -> detect -> embed -> cluster -> organize)
        -> visage/scanner.py (scan_images - discover photos)
        -> visage/detector.py + detectors/ (Vision, SCRFD, YuNet backends)
        -> visage/embedder.py + embedding/ (dlib, InsightFace backends, GPU service)
        -> visage/cluster/ (HDBSCAN/DBSCAN, incremental, optimizer)
        -> visage/organizer.py (build_organize_plan, execute_organize_plan)

Web UI (visage --web)
  -> visage/serve.py (FastAPI + uvicorn)
    -> visage/server/app.py (FastAPI app)
    -> visage/server/routes.py + routes_*.py (REST API)
    -> frontend/ (React 19 + TypeScript + Vite + Tailwind)

Desktop App (Tauri)
  -> src-tauri/ (Rust + Tauri 2, bundles frontend/dist into native macOS app)
  -> frontend/ (built and embedded via Tauri)
```

## Critical Files Map

| File | Role |
|------|------|
| `src/visage/cli.py` (12KB) | CLI entry point -- argparse command-line interface |
| `src/visage/pipeline.py` (16KB) | Core pipeline orchestration -- scan/detect/embed/cluster/organize |
| `src/visage/config.py` (11KB) | VisageConfig dataclass and config builder with hardware detection |
| `src/visage/detectors/vision.py` (15KB) | macOS Vision framework face detection backend |
| `src/visage/detectors/__init__.py` (4.5KB) | Detector backend registry and auto-selection |
| `src/visage/embedding/service.py` (9KB) | visage-engine embedding service (standalone process) |
| `src/visage/cluster/core.py` (23KB) | HDBSCAN/DBSCAN clustering with confidence scoring |
| `src/visage/cluster/incremental.py` (5KB) | Incremental clustering for streaming/new photos |
| `src/visage/cache.py` (9KB) | Embedding cache -- stores face vectors to avoid recomputation |
| `src/visage/organizer.py` (7KB) | File organization -- generate and execute copy/move plans |
| `src/visage/server/app.py` (11KB) | FastAPI web application |
| `src/visage/server/workspace.py` (34KB) | Workspace management for web UI |
| `src/visage/server/routes.py` (19KB) | Main REST API routes |
| `src/visage/classify/clip_model.py` (13KB) | CLIP-based photo classification |
| `src/visage/quality/core.py` (7KB) | Face quality assessment |
| `src/visage/active/prototype.py` (6KB) | Active learning prototype selection |
| `src/visage/events/` | Event clustering, cover selection, naming, timeline |
| `src/visage/vector/index.py` (10KB) | FAISS vector index for fast similarity search |
| `src/visage/library/manager.py` (6KB) | Photo library management |
| `frontend/src/` | React 19 SPA with @tanstack/react-query, zustand, Tailwind |
| `frontend/vite.config.ts` | Vite build configuration |
| `src-tauri/tauri.conf.json` | Tauri desktop app configuration |
| `src-tauri/Cargo.toml` | Rust dependencies (Tauri 2, reqwest, tokio) |
| `docs/architecture.md` | Architecture documentation |
| `docs/configuration.md` | Configuration reference |
| `docs/development.md` | Development guide |
| `tests/` | 31 test files covering all subsystems |

## Development Rules

1. **Python version**: >= 3.10, use `from __future__ import annotations` throughout
2. **Build system**: setuptools, not Poetry. Package in `src/visage/`, not flat.
3. **Dependencies**: Use `uv` for package management. `uv.lock` is committed.
4. **Linting**: ruff with E, F, I, B, UP rules. Line length 100.
5. **Testing**: pytest with -ra flags. Tests in `tests/` mirror `src/visage/` structure.
6. **Type annotations**: Use full type hints. Data classes prefer `@dataclass`.
7. **Logging**: Use `logging.getLogger(__name__)` throughout, not print.
8. **Imports**: Use absolute imports from `visage.*`. Run from repo root.
9. **Face detection backends**: "auto" detection selects best available (Vision on macOS, SCRFD fallback). All backends implement DetectorBackend interface.
10. **Embedding backends**: dlib (face_recognition) is default, InsightFace is preferred when available. Embedding service runs as standalone process (`visage-engine`).
11. **Tauri**: Frontend is built first (`cd frontend && npm run build`), then Tauri bundles `frontend/dist/`. See `src-tauri/tauri.conf.json` for build commands.
12. **Security**: `.claude/settings.json` enforces `acceptEdits` mode -- do not weaken deny rules without review.

## Sub-Packages Quick Reference

| Package | Purpose |
|---------|---------|
| `visage.active` | Active learning: prototype selection, correction store, nearest centroid |
| `visage.batch` | Batch processing: checkpoint, queue |
| `visage.classify` | Photo classification: CLIP model, scene detection, tag store |
| `visage.cluster` | Face clustering: HDBSCAN/DBSCAN core, incremental, optimizer |
| `visage.core` | Core re-exports and utilities |
| `visage.db` | Database migrations (SQL) |
| `visage.detectors` | Face detection backends: Vision, SCRFD, YuNet, NMS |
| `visage.embedding` | Embedding service: backend, batcher, GPU detection |
| `visage.ensemble` | Ensemble classifier combining multiple models |
| `visage.events` | Event detection: cluster by time/location, cover selection, naming |
| `visage.library` | Photo library management |
| `visage.quality` | Face quality scoring |
| `visage.server` | Web API: FastAPI app, routes, search, workspace |
| `visage.vector` | FAISS vector index and metadata store |

## Avoid

- Hardcoding credentials or API keys
- Adding dependencies without clear justification
- Using `rm -rf`, `sudo`, `chown` (blocked by deny rules)
- Committing `.DS_Store`, `__pycache__`, or build artifacts
- Modifying Tauri build config without understanding frontend build pipeline
- Skipping type hints or using implicit Optional
