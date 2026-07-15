# duckdb_turbovec

DuckDB extension for compressed vector search using Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) via the [turbovec](https://github.com/RyanCodrai/turbovec) crate.

**31 GB → 4 GB. Faster than FAISS on ARM. Zero training.**

```sql
-- Build index (Python)
--   idx = TurboQuantIndex(dim=1536, bit_width=4)
--   idx.add(vectors)
--   idx.write('myidx.tv')

-- Search in DuckDB
LOAD 'turbovec.duckdb_extension';
SELECT * FROM turboquant_search('myidx.tv', ?::FLOAT[1536], 10);
-- Returns: (idx INT, score FLOAT)
```

## Features

| | |
|---|---|
| Compression | 2-bit (16×) / 4-bit (8×) — data-oblivious, no training |
| Search | SIMD-accelerated: NEON (ARM), AVX-512BW/AVX2 (x86) |
| Recall | Beats FAISS PQ: +0.2–1.9pp R@1 on OpenAI embeddings |
| Speed | ARM: +12–20% vs FAISS FastScan; x86: wins 4-bit, ties 2-bit |
| Integration | Pure Rust, single `.duckdb_extension` file |

## Install

```bash
# From local build
git clone https://github.com/alitrack/duckdb_turbovec
cd duckdb_turbovec
make release
# → build/release/turbovec.duckdb_extension
```

Requires: Rust, DuckDB ≥ v1.0, `libopenblas-dev` (Linux) / Accelerate (macOS).

## Usage

```sql
INSTALL turbovec;
LOAD turbovec;

-- Search a pre-built turbovec index
SELECT * FROM turboquant_search(
    '/path/to/index.tv',
    [0.1, 0.2, ...]::FLOAT[1536],
    10
);
```

**Index building** currently requires Python `turbovec`:

```python
from turbovec import TurboQuantIndex
import numpy as np

idx = TurboQuantIndex(dim=1536, bit_width=4)
idx.add(embeddings)  # numpy float32, shape [N, 1536]
idx.write('myidx.tv')
```

## Roadmap

- [ ] `turboquant_encode()` / `turboquant_score()` scalar functions (store compressed vectors in BLOBs)
- [ ] `turboquant_build()` — build index from DuckDB table (PRAGMA or table function)
- [ ] DuckDB Community Extension submission
- [ ] Multi-platform CI (Linux x86_64, macOS ARM, macOS x86_64)

## License

MIT — matches [turbovec](https://github.com/RyanCodrai/turbovec) (MIT) and duckdb-rs (MIT).
