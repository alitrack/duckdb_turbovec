# duckdb_turbovec

DuckDB extension for compressed vector search using Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) via the [turbovec](https://github.com/RyanCodrai/turbovec) crate.

**31 GB → 4 GB. Faster than FAISS on ARM. Zero training. Pure DuckDB — no Python required.**

> **Score semantics:** All scores are approximate quantized similarity values, NOT exact cosine distances. TurboQuant is a lossy compression (8-16×). Scores are comparable within the same index, but not directly comparable to exact cosine scores from other engines.

## Quickstart

```sql
LOAD 'turbovec.duckdb_extension';

-- Method 1: Build from table via SET VARIABLE
CREATE TABLE documents(id INTEGER, emb FLOAT[1536]);
INSERT INTO documents VALUES (1, [0.1, 0.2, ...]), (2, [0.2, 0.1, ...]);
SET VARIABLE vec_str = (SELECT string_agg(emb::VARCHAR, ',') FROM documents);
SELECT * FROM turboquant_build(getvariable('vec_str'), 1536, 4, '/tmp/myidx.tv');

-- Method 2: Build from literal LIST (dim auto-detected)
SELECT * FROM turboquant_build_list(
  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,  0.2,0.1,0.4,0.3,0.6,0.5,0.8,0.7],
  4, '/tmp/myidx.tv'
);

-- Method 3: IVF index (K-means partitions for sub-linear search at scale)
SELECT * FROM turboquant_build_ivf(
  getvariable('vec_str'), 1536, 4, 256, '/tmp/myivf/'
);
-- num_lists = 0 → auto (sqrt(n)). IVF v2 stores all clusters in ONE
-- index file (index.tv + meta2.bin) instead of one .tv per cluster —
-- 18x smaller at 1.5k docs, 7x at 100k. Old per-cluster indexes still
-- searchable (format auto-detected).

-- Top-k search
SELECT * FROM turboquant_search('/tmp/myidx.tv', '[0.1, 0.2, ...]', 10);

-- IVF search (with probe count)
SELECT * FROM turboquant_search_ivf('/tmp/myivf/', '[0.1, 0.2, ...]', 10, 8);
-- Loaded indexes are cached process-wide: repeated queries skip the
-- file read (21x faster on a 100k-vector index).

-- Rerank (oversampling + exact rescoring) — recall recovery to ~exact.
-- Build with keep_raw=true (6th param) to store the original f32 vectors
-- next to the quantized index, then search with rerank_factor (5th param):
-- the quantized index proposes k x factor candidates, exact f32 distances
-- settle the final top-k (the Qdrant/VectoGrid pattern).
SELECT * FROM turboquant_build_ivf(
  getvariable('vec_str'), 1536, 4, 0, '/tmp/myivf_rr/', true
);
SELECT * FROM turboquant_search_ivf('/tmp/myivf_rr/', '[0.1, ...]', 10, 32, 4);
-- recall@10 on 1472x1536 OpenAI embeddings: plain 0.846 -> 0.996 @ ~2 ms
-- index with raw vectors: 12.1 MB (still 2.3x smaller than DuckIR u8)

-- Recall recovery: ensemble of m permutation-decorrelated indexes,
-- score-averaged at search time (no native 8-bit in the kernel; this
-- trades m x memory for recall instead)
SELECT * FROM turboquant_build_ensemble(
  getvariable('vec_str'), 1536, 4, 4, '/tmp/myens/'
);
SELECT * FROM turboquant_search_ensemble('/tmp/myens/', '[0.1, ...]', 10);
-- recall@10 on 1472x1536 OpenAI embeddings: single 0.888 -> m=4 0.939
-- → (idx INT, score FLOAT) — top 10 nearest neighbors

-- Full scoring — all vectors ranked by similarity
SELECT * FROM turboquant_score('/tmp/myidx.tv', '[0.1, 0.2, ...]');
-- → (idx INT, score FLOAT) — every vector in the index, sorted by score DESC

-- Radius (WITHIN) search — everything within `eps` of the nearest, optional k cap
SELECT * FROM turboquant_search_within('/tmp/myidx.tv', '[0.1, 0.2, ...]', 0.05, 0);
-- → (idx INT, score FLOAT) — all vectors whose score is within eps of the top-1,
--   sorted best-first; k=0 = no cap, k>0 = top-k fused with the radius bound
```

## SQL API

| Function | Parameters | Returns |
|---|---|---|
| `turboquant_build(vectors_str, dim, bit_width, output_path)` | nested array string `[[a,b],[c,d]]`, dim, 2/3/4, path | (output_path, rows) |
| `turboquant_build_list(vectors_list, bit_width, output_path)` | flat `[f1,f2,...]` LIST, 2/3/4, path | (output_path, rows, dim) — dim auto-detected |
| `turboquant_build_concat(output_path, dim, bit_width, values_str)` | path, dim, 2/3/4, comma-separated `f1,f2,...` | (output_path, rows) |
| `turboquant_build_ivf(vectors_str, dim, bit_width, num_lists, output_dir, [keep_raw])` | nested array string, dim, 2/3/4, k (0=auto √n), dir, keep raw f32 for rerank | (output_dir, rows, num_lists) |
| `turboquant_add(index_path, vectors_str, dim)` | existing .tv path, nested array string, dim | (output_path, added, total) |
| `turboquant_remove(index_path, idx)` | .tv file path, vector index to remove | (output_path, removed_idx, remaining) |
| `turboquant_search(index_path, query_str, k)` | .tv file path, query `[f1,f2,...]`, top-k | (idx, score) |
| `turboquant_search_within(index_path, query_str, eps, k)` | .tv file path, query `[f1,f2,...]`, score-epsilon, cap (0=none) | (idx, score) — all within `eps` of nearest, best-first; k>0 = top-k+radius fused |
| `turboquant_search_ivf(index_dir, query_str, k, probes, rerank_factor)` | IVF dir path, query string, top-k, probes (**0 = probe every cluster**), rerank factor (0/1 = plain quantized search). ⚠ `rerank_factor` is documented as optional but **must be passed** — DuckDB's binder rejects the 4-arg call (`No function matches ...`, see quack-rs arity note) | (idx, score) |
| `turboquant_search_ivf_filter(index_dir, query_str, k, filter_ids, [max_probes := ], [rerank_factor := ])` | IVF dir path (V2), query string, top-k, `LIST<BIGINT>` of ids passing an arbitrary SQL predicate, optional NAMED probe cap / rerank factor (default from `turbovec_config()` knobs) | (idx, score) — filtered vector search with iterative probe widening |
| `turbovec_config('knob', value)` / `turbovec_config('knob')` | Set/get process-wide search defaults: `ivf_default_max_probes`, `ivf_default_rerank_factor`. `turbovec_config('', NULL)` resets all. Explicit function args always win | BIGINT (echoed value / current setting; NULL = unset or unknown knob) |
| `turboquant_build_ensemble(vectors_str, dim, bit_width, members, output_dir)` | nested array string, dim, 2/3/4, m (1–8), dir | (output_dir, rows, members) |
| `turboquant_search_ensemble(ensemble_dir, query_str, k)` | ensemble dir path, query string, top-k | (idx, score) — score-averaged across members |
| `turboquant_score(index_path, query_str)` | .tv file path, query `[f1,f2,...]` | (idx, score) — all vectors, sorted DESC |

> **Note:** DuckDB table functions do not accept subquery parameters. Use `SET VARIABLE` + `getvariable()` to pass computed values from tables (see Quickstart Method 1).

## Build

```bash
git clone git@github.com:alitrack/duckdb_turbovec.git && cd duckdb_turbovec

# macOS (uses Accelerate framework)
cargo build --release
python3 scripts/metadata.py target/release/libturbovec.dylib -o turbovec.duckdb_extension

# Linux (requires libopenblas-dev)
sudo apt-get install libopenblas-dev
cargo build --release
python3 scripts/metadata.py target/release/libturbovec.so -o turbovec.duckdb_extension

# Windows (no BLAS dependency — turbovec uses pure Rust on Windows)
cargo build --release
python3 scripts/metadata.py target/release/turbovec.dll -o turbovec.duckdb_extension
```

> **WASM**: Not supported — turbovec requires 64-bit pointer width; WASM is 32-bit.

## Running Tests

```bash
bash tests/integration_test.sh [path_to_duckdb_cli]
```

Requires DuckDB CLI and `turbovec.duckdb_extension` built in the project root.

## Features

- 2-bit (16×) / 4-bit (8×) compression — data-oblivious, zero training
- SIMD search: NEON (ARM), AVX-512BW/AVX2 (x86)
- Beats FAISS PQ: +0.2–1.9pp R@1 on OpenAI embeddings
- ARM: +12–20% vs FAISS FastScan
- Two build modes: string-based (`turboquant_build`), list-based (`turboquant_build_list`), flat CSV (`turboquant_build_concat`), and IVF (`turboquant_build_ivf`)
- Table-sourced builds via `SET VARIABLE` + `getvariable()`
- IVF index with K-means partitioning for sub-linear search at scale
- IVF v2 single-file layout — one index for all clusters, 7–18× smaller than per-cluster files
- k-means++ seeding + parallel Lloyd assignment (`num_lists=0` → auto √n)
- Rerank: `keep_raw` build + `rerank_factor` search — recall@10 0.846 → 0.996 on real OpenAI embeddings (~2 ms)
- Ensemble indexes (m members, score-averaged) — recall@10 0.888 → 0.939 at ×4
- Process-wide index cache — repeated search on a 100k index 37.7 ms → 1.77 ms
- Incremental add/remove (`turboquant_add` / `turboquant_remove`)
- IVF auto-probe (`probes=0` → full scan)
- Radius (WITHIN) search (`turboquant_search_within`) — everything within `eps` of the nearest, optional top-k cap

## Roadmap

- [x] `turboquant_search()` VTab
- [x] `turboquant_build()` — build index from table via `SET VARIABLE` + `string_agg`
- [x] `turboquant_score()` — all-vector scoring
- [x] `turboquant_build_list()` — build from flat `LIST<FLOAT>` with auto dim detection
- [x] `turboquant_build_ivf()` / `turboquant_search_ivf()` — IVF index with K-means routing
- [x] `turboquant_search_within()` — radius (WITHIN) / top-k+eps fused search
- [x] IVF v2 single-file layout + k-means++ + parallel build
- [x] Process-wide index cache
- [x] Rerank (`keep_raw` + `rerank_factor`) — recall to ~exact
- [x] Ensemble indexes — recall knob without native 8-bit
- [x] Filtered vector search (`turboquant_search_ivf_filter`) — arbitrary SQL predicates as a `LIST<BIGINT>` allowlist, iterative probe widening (pgvector 0.8 iterative-scan semantics, minus the HNSW filtered-recall pathology), optional rerank
- [x] macOS ARM + Linux x86_64 + Windows x86_64 CI
- [x] Integration test suite (26 tests)
- [ ] DuckDB Community Extension submission (PR #3623 open, CI green)

## Known Limitations

- DuckDB table functions reject subquery parameters. Use `SET VARIABLE name = (...)` + `getvariable('name')` to pass computed values.
- Scores are quantized approximations, not exact cosine. Self-vectors may not score highest for very small indices.
- `turboquant_search_within` `eps` is a **score gap in the index's own quantized units** (not a cosine/distance distance). Calibrate it per index from the score spread (e.g. via `turboquant_score`); it is not comparable across engines or metrics.
- IVF indices are immutable at the cluster level (no per-cluster add/remove). Use flat index `turboquant_add`/`turboquant_remove` for incremental CRUD.

## License

MIT — matches turbovec (MIT) and duckdb-rs (MIT).
