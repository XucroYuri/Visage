# Visage Performance Benchmarks

Benchmark results are recorded here automatically by `scripts/benchmark.sh`.

## Baseline (Phase 1 Start)

| Metric | Target |
|--------|--------|
| Detection recall improvement | >20% over baseline |
| Clustering purity | >0.95 |
| Per-image detection time | No significant increase |
| 10K image pipeline total time | No significant increase |
| Tauri idle memory | <200MB |

## How to Run

```bash
# With your own photo directory:
./scripts/benchmark.sh /path/to/photos

# With synthetic test data (100 images by default):
./scripts/benchmark.sh

# Custom count:
./scripts/benchmark.sh --count 1000
```
