---
name: duckdb-turbovec-ann
description: >
  Build and search compressed vector indexes in DuckDB SQL with the turbovec
  extension (TurboQuant, ICLR 2026): flat indexes, IVF indexes for
  sub-linear search at scale, incremental add/remove, top-k ANN queries.
  Use for semantic search over embeddings at millions-of-vectors scale with
  8-16x memory compression and no Python/FAISS.
version: 0.3.3
---

# turbovec: compressed vector search

Extension: `turbovec` (Google TurboQuant via the turbovec Rust crate).
`INSTALL turbovec FROM community; LOAD turbovec;`
31 GB → 4 GB. Zero training. Faster than FAISS on ARM.

**Score semantics:** scores are approximate quantized similarities, NOT exact
cosine distances (lossy 8-16× compression). Comparable within one index;
don't compare across engines.

## Build

```sql
-- From a table (SET VARIABLE bridges table data into table-function args)
CREATE TABLE documents(id INTEGER, emb FLOAT[1536]);
SET VARIABLE vec_str = (SELECT string_agg(emb::VARCHAR, ',') FROM documents);
SELECT * FROM turboquant_build(getvariable('vec_str'), 1536, 4, '/tmp/myidx.tv');

-- From a literal LIST (dim auto-detected from flat length)
SELECT * FROM turboquant_build_list(
  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,  0.2,0.1,0.4,0.3,0.6,0.5,0.8,0.7],
  4, '/tmp/myidx.tv');                       -- 4 = bit_width (2/3/4)

-- IVF index: K-means partitions for sub-linear search at scale
SELECT * FROM turboquant_build_ivf(
  getvariable('vec_str'), 1536, 4, 256, '/tmp/myivf/');   -- 256 lists → dir
```

Higher bit_width = better recall, bigger index. Also
`turboquant_build_concat(path, dim, bit_width, 'f1,f2,...')` for
comma-separated values.

## Search

```sql
-- Top-k flat
SELECT * FROM turboquant_search('/tmp/myidx.tv', '[0.1, 0.2, ...]', 10);
-- → (idx INT, score FLOAT)

-- IVF with probe count (0 = all lists; raise for recall, lower for speed)
SELECT * FROM turboquant_search_ivf('/tmp/myivf/', '[0.1, 0.2, ...]', 10, 8);

-- Full ranking (every vector, sorted by score DESC) — for evaluation
SELECT * FROM turboquant_score('/tmp/myidx.tv', '[0.1, 0.2, ...]');
```

## Update in place

```sql
SELECT * FROM turboquant_add('/tmp/myidx.tv', '[[...new vectors...]]', 1536);
SELECT * FROM turboquant_remove('/tmp/myidx.tv', 42);   -- by vector index
```

## Gotchas

- DuckDB table functions reject subquery parameters — always
  `SET VARIABLE x = (SELECT ...)` then `getvariable('x')`.
- Map result `idx` back to rows yourself: keep a side table
  `rowid ↔ index position` (build order defines idx).
- Exact (<100k rows) needs? Use the `ml` extension's
  `ml_similarity_value` full-scan cosine instead.
