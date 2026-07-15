# duckdb_turbovec

DuckDB extension for compressed vector search using Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) via the [turbovec](https://github.com/RyanCodrai/turbovec) crate.

**31 GB → 4 GB. Faster than FAISS on ARM. Zero training. Pure DuckDB — no Python turbovec required.**

## Quickstart

```sql
LOAD 'turbovec.duckdb_extension';

-- 1. Serialize and build index from a DuckDB table
--    First get vectors as string, then feed to turboquant_build
SELECT * FROM turboquant_build(
    (SELECT string_agg(emb::VARCHAR, ',') FROM documents), 1536, 4, '/tmp/myidx.tv'
);

-- 2. Top-k search
SELECT * FROM turboquant_search('/tmp/myidx.tv', '[0.1, 0.2, ...]', 10);
-- → (idx INT, score FLOAT) — top 10 nearest neighbors

-- 3. Full scoring — all vectors with similarity scores
SELECT * FROM turboquant_score('/tmp/myidx.tv', '[0.1, 0.2, ...]');
-- → (idx INT, score FLOAT) — every vector in the index, sorted by score DESC
```

## SQL API

| Function | Parameters | Returns |
|---|---|---|
| `turboquant_build(vectors_str, dim, bit_width, output_path)` | vectors_str: `[arr],[arr],...` from `string_agg`, dim, 2\|4, path | (path, rows) |
| `turboquant_search(index_path, query_str, k)` | .tv file path, query as `'[f1,f2,...]'` string, top-k | (idx, score) — top-k |
| `turboquant_score(index_path, query_str)` | .tv file path, query as `'[f1,f2,...]'` string | (idx, score) — all vectors, sorted by score DESC |

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

## Features

- 2-bit (16×) / 4-bit (8×) compression — data-oblivious, zero training
- SIMD search: NEON (ARM), AVX-512BW/AVX2 (x86)
- Beats FAISS PQ: +0.2–1.9pp R@1 on OpenAI embeddings
- ARM: +12–20% vs FAISS FastScan

## Roadmap

- [x] `turboquant_search()` VTab
- [x] `turboquant_build()` — build index from DuckDB table via `string_agg`
- [x] `turboquant_score()` — all-vector scoring
- [x] macOS ARM + Linux x86_64 CI
- [ ] DuckDB Community Extension submission

## License

MIT — matches turbovec (MIT) and duckdb-rs (MIT).
