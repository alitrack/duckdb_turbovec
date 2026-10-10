# duckdb-turbovec — Claude Code Plugin

One skill giving your AI assistant verified knowledge of the
[duckdb_turbovec](https://github.com/alitrack/duckdb_turbovec) extension:
TurboQuant compressed ANN indexes (flat + IVF), search/score/add/remove,
bit_width tradeoffs, and score semantics caveats.

## Install

```text
/plugin marketplace add alitrack/duckdb_turbovec
/plugin install duckdb-turbovec@duckdb-turbovec
```

## Skills

| Skill | Covers |
|---|---|
| `duckdb-turbovec-ann` | build/build_ivf/search/score/add/remove, probe tuning, idx mapping |

```sql
INSTALL turbovec FROM community; LOAD turbovec;
```

MIT, alitrack.
