# duckdb_turbovec

DuckDB extension for compressed vector search using Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) via the [turbovec](https://github.com/RyanCodrai/turbovec) crate.

**31 GB → 4 GB. Faster than FAISS on ARM. Zero training.**

## Quickstart

```bash
# 1. Build index (Python)
pip install turbovec duckdb
```

```python
import numpy as np
from turbovec import TurboQuantIndex
import duckdb

# Build and save index
idx = TurboQuantIndex(dim=1536, bit_width=4)
idx.add(embeddings)  # numpy float32, shape [N, 1536]
idx.write("/tmp/myidx.tv")

# Load extension and search
con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.load_extension("turbovec.duckdb_extension")

query_str = str([float(x) for x in query_vector])
result = con.execute(f"""
    SELECT * FROM turboquant_search('/tmp/myidx.tv', '{query_str}', 10)
""").fetchall()
# → [(idx, score), ...]
```

## SQL API

```sql
LOAD 'turbovec.duckdb_extension';

-- turboquant_search(index_path VARCHAR, query_str VARCHAR, k INTEGER) → TABLE(idx INT, score FLOAT)
SELECT * FROM turboquant_search('/path/to/index.tv', '[0.1, 0.2, ...]', 10);
```

The query vector is passed as a VARCHAR containing a Python/Rust-style float array string `'[1.0, 2.0, ...]'`.

## Features

| | |
|---|---|
| Compression | 2-bit (16×) / 4-bit (8×) — data-oblivious, no training |
| Search | SIMD-accelerated: NEON (ARM), AVX-512BW/AVX2 (x86) |
| Recall | Beats FAISS PQ: +0.2–1.9pp R@1 on OpenAI embeddings |
| Speed | ARM: +12–20% vs FAISS FastScan; x86: wins 4-bit, ties 2-bit |
| Integration | Pure Rust, single `.duckdb_extension` file |

## Build from Source

```bash
git clone git@github.com:alitrack/duckdb_turbovec.git
cd duckdb_turbovec

# macOS (uses Accelerate framework)
cargo build --release
python3 scripts/metadata.py target/release/libturbovec.dylib -o turbovec.duckdb_extension

# Linux (requires libopenblas-dev)
sudo apt-get install libopenblas-dev
cargo build --release
python3 scripts/metadata.py target/release/libturbovec.so -o turbovec.duckdb_extension
```

Requires: Rust, DuckDB ≥ v1.0.

## Roadmap

- [x] `turboquant_search()` — VTab for searching pre-built `.tv` indexes
- [x] macOS ARM + Linux x86_64 CI
- [ ] `turboquant_build_from_table()` — build index from DuckDB table (requires duckdb-rs Connection access)
- [ ] DuckDB Community Extension submission
- [ ] Multi-platform release artifacts (macOS x86_64, Windows)

## License

MIT — matches [turbovec](https://github.com/RyanCodrai/turbovec) (MIT) and duckdb-rs (MIT).
