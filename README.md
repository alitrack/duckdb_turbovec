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

-- Top-k search
SELECT * FROM turboquant_search('/tmp/myidx.tv', '[0.1, 0.2, ...]', 10);

-- IVF search (with probe count)
SELECT * FROM turboquant_search_ivf('/tmp/myivf/', '[0.1, 0.2, ...]', 10, 8);
-- → (idx INT, score FLOAT) — top 10 nearest neighbors

-- Full scoring — all vectors ranked by similarity
SELECT * FROM turboquant_score('/tmp/myidx.tv', '[0.1, 0.2, ...]');
-- → (idx INT, score FLOAT) — every vector in the index, sorted by score DESC
```

## SQL API

| Function | Parameters | Returns |
|---|---|---|---|
| `turboquant_build(vectors_str, dim, bit_width, output_path)` | vectors_str: `'[[arr],[arr],...'` string, dim, 2\|3\|4, path | (output_path, rows) |
| `turboquant_build_list(vectors_list, bit_width, output_path)` | vectors_list: flat `[f1,f2,...]` LIST, 2\|3\|4, path | (output_path, rows, dim) — dim auto-detected |
| `turboquant_build_concat(output_path, dim, bit_width, values_str)` | path, dim, 2\|3\|4, comma-separated `'f1,f2,...'` | (output_path, rows) |
| `turboquant_build_ivf(vectors_str, dim, bit_width, num_lists, output_dir)` | vectors_str, dim, 2\|3\|4, num_lists (K-means k), output_dir | (output_dir, rows, num_lists) |
| `turboquant_add(index_path, vectors_str, dim)` | existing .tv path, new vectors string, dim | (output_path, added, total) |
| `turboquant_remove(index_path, idx)` | .tv file path, vector index to remove | (output_path, removed_idx, remaining) |
| `turboquant_search(index_path, query_str, k)` | .tv file path, query as `'[f1,f2,...]'` string, top-k | (idx, score) |
| `turboquant_search_ivf(index_dir, query_str, k, probes)` | IVF dir path, query string, top-k, num probes (0=all) | (idx, score) |
| `turboquant_score(index_path, query_str)` | .tv file path, query as `'[f1,f2,...]'` string | (idx, score) — all vectors, sorted DESC |

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
- Incremental add/remove (`turboquant_add` / `turboquant_remove`)
- IVF auto-probe (`probes=0` → full scan)

## Roadmap

- [x] `turboquant_search()` VTab
- [x] `turboquant_build()` — build index from table via `SET VARIABLE` + `string_agg`
- [x] `turboquant_score()` — all-vector scoring
- [x] `turboquant_build_list()` — build from flat `LIST<FLOAT>` with auto dim detection
- [x] `turboquant_build_ivf()` / `turboquant_search_ivf()` — IVF index with K-means routing
- [x] macOS ARM + Linux x86_64 + Windows x86_64 CI
- [x] Integration test suite (16 tests)
- [ ] DuckDB Community Extension submission

## Known Limitations

- DuckDB table functions reject subquery parameters. Use `SET VARIABLE name = (...)` + `getvariable('name')` to pass computed values.
- Scores are quantized approximations, not exact cosine. Self-vectors may not score highest for very small indices.
- IVF indices are immutable at the cluster level (no per-cluster add/remove). Use flat index `turboquant_add`/`turboquant_remove` for incremental CRUD.

## License

MIT — matches turbovec (MIT) and duckdb-rs (MIT).
