# Changelog

## [0.2.0] — 2026-07-24

### Added
- `turboquant_build_list(vectors_list, bit_width, output_path)` — build from flat `LIST<FLOAT>` with auto-dim detection
- `turboquant_build_ivf(vectors_str, dim, bit_width, num_lists, output_dir)` — IVF index with K-means partitioning
- `turboquant_search_ivf(index_dir, query_str, k, probes)` — IVF search with probe routing
- Per-cluster global ID mapping (`idmap_*.bin`) for IVF search
- Integration test suite: 16 tests using DuckDB CLI
- `SET VARIABLE` + `getvariable()` workaround for DuckDB table function subquery limitation

### Fixed
- IVF search now returns global vector indices (was cluster-local)

## [0.1.0] — 2026-07-21

### Added
- `turboquant_search(index_path, query_str, k)` — top-k vector search
- `turboquant_score(index_path, query_str)` — all-vector scoring
- `turboquant_build(vectors_str, dim, bit_width, output_path)` — build from nested array string
- 2-bit (16×) and 4-bit (8×) compression via turbovec crate
- macOS ARM, Linux x86_64, Windows x86_64 CI workflows
- `scripts/metadata.py` — DuckDB extension packaging
