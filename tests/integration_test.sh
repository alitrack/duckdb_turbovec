#!/usr/bin/env bash
# Integration tests for duckdb_turbovec extension
# Usage: ./tests/integration_test.sh [path_to_duckdb_cli]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DUCKDB="${1:-duckdb} -unsigned"
TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}FAIL${NC} $1 — $2"; FAIL=$((FAIL+1)); }

echo "=== duckdb_turbovec Integration Tests ==="
echo "Extension: $PROJECT_DIR/turbovec.duckdb_extension"
echo "DuckDB:    $($DUCKDB --version)"
echo ""

# ── Test: extension loads ──
SQL="LOAD '$PROJECT_DIR/turbovec.duckdb_extension'; SELECT 'loaded';"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1) && pass "extension loads" || fail "extension loads" "$OUT"

# ── Test: turboquant_build ──
IDX="$TMPDIR/test_build.tv"
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build('[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],[0.2,0.1,0.4,0.3,0.6,0.5,0.8,0.7]]', 8, 4, '$IDX');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "$IDX" && pass "turboquant_build creates index" || fail "turboquant_build" "$OUT"
echo "$OUT" | grep -q "2" && pass "turboquant_build reports 2 rows" || fail "turboquant_build rows" "$OUT"

# ── Test: turboquant_search ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT idx FROM turboquant_search('$IDX', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 2);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "0" && pass "turboquant_search returns idx=0" || fail "turboquant_search" "$OUT"
[ "$(echo "$OUT" | wc -l)" -ge 2 ] && pass "turboquant_search returns 2 results" || fail "turboquant_search count" "$OUT"

# ── Test: turboquant_score ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_score('$IDX', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "2" && pass "turboquant_score returns all 2 vectors" || fail "turboquant_score" "$OUT"

# ── Test: k > index size (clamped) ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search('$IDX', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 999);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "2" && pass "k > n clamped" || fail "clamp" "$OUT"

# ── Test: 3-bit quantization ──
IDX3="$TMPDIR/test_3bit.tv"
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build('[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]', 8, 3, '$IDX3');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "$IDX3" && pass "3-bit quantization" || fail "3-bit" "$OUT"

# ── Test: k=0 returns empty ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search('$IDX', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 0);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "0" && pass "k=0 returns empty" || fail "k=0" "$OUT"

# ── Test: dim mismatch ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build('[[0.1,0.2]]', 8, 4, '$TMPDIR/bad.tv');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1) || true
echo "$OUT" | grep -qi "dim\|mismatch\|expected" && pass "dim mismatch error" || fail "dim mismatch" "$OUT"

# ── Test: from table via SET VARIABLE ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
CREATE OR REPLACE TABLE test_vecs AS SELECT * FROM (VALUES
  ([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]::FLOAT[8]),
  ([0.2,0.1,0.4,0.3,0.6,0.5,0.8,0.7]::FLOAT[8]),
  ([0.3,0.2,0.5,0.4,0.7,0.6,0.9,0.8]::FLOAT[8])
) t(emb);
SET VARIABLE vec_str = (SELECT string_agg(emb::VARCHAR, ',') FROM test_vecs);
SELECT * FROM turboquant_build(getvariable('vec_str'), 8, 4, '$TMPDIR/from_table.tv');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "3" && pass "build from table (3 vectors)" || fail "from table" "$OUT"

# ── Test: search table-built index ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search('$TMPDIR/from_table.tv', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 1);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "1" && pass "search from table index" || fail "search table" "$OUT"

# ── Test: turboquant_build_list ──
IDX_LIST="$TMPDIR/test_list.tv"
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build_list(
  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,  0.2,0.1,0.4,0.3,0.6,0.5,0.8,0.7,  0.3,0.2,0.5,0.4,0.7,0.6,0.9,0.8],
  4, '$IDX_LIST'
);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "3" && pass "build_list LIST input (3 vectors)" || fail "build_list" "$OUT"

# ── Test: search LIST-built index ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search('$IDX_LIST', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 2);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "2" && pass "search from LIST index" || fail "search list" "$OUT"

# ── Test: IVF build ──
IVF_DIR="$TMPDIR/ivf_test"
VECS='[[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80],[0.11,0.21,0.31,0.41,0.51,0.61,0.71,0.81],[0.12,0.22,0.32,0.42,0.52,0.62,0.72,0.82],[0.13,0.23,0.33,0.43,0.53,0.63,0.73,0.83],[0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50],[0.51,0.51,0.51,0.51,0.51,0.51,0.51,0.51],[0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52],[0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53],[0.90,0.90,0.90,0.90,0.90,0.90,0.90,0.90],[0.91,0.91,0.91,0.91,0.91,0.91,0.91,0.91],[0.92,0.92,0.92,0.92,0.92,0.92,0.92,0.92],[0.93,0.93,0.93,0.93,0.93,0.93,0.93,0.93]]'
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build_ivf('${VECS}', 8, 4, 3, '$IVF_DIR');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "12" && pass "IVF build 12 vectors" || fail "IVF build" "$OUT"

# ── Test: IVF search ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search_ivf('$IVF_DIR', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 5, 3);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "5" && pass "IVF search top-5" || fail "IVF search" "$OUT"

# ── Test: IVF auto-probe (probes=0 → full scan) ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search_ivf('$IVF_DIR', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]', 5, 0);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "5" && pass "IVF auto-probe returns top-5" || fail "IVF auto-probe" "$OUT"

# ── Test: turboquant_add (incremental append) ──
IDX_ADD="$TMPDIR/add_test.tv"
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build('[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],[0.2,0.1,0.4,0.3,0.6,0.5,0.8,0.7]]', 8, 4, '$IDX_ADD');
"
$DUCKDB -noheader -c "$SQL" 2>&1 > /dev/null
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_add('$IDX_ADD', '[[0.3,0.1,0.2,0.4,0.6,0.5,0.7,0.8]]', 8);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "3" && pass "turboquant_add total=3" || fail "turboquant_add" "$OUT"

# ── Test: turboquant_remove ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_remove('$IDX_ADD', 0);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "2" && pass "turboquant_remove remaining=2" || fail "turboquant_remove" "$OUT"

# ── Test: turboquant_build_concat ──
IDX_CONCAT="$TMPDIR/concat_test.tv"
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT * FROM turboquant_build_concat('$IDX_CONCAT', 4, 4, '0.1,0.2,0.3,0.4,0.2,0.1,0.4,0.3');
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "2" && pass "turboquant_build_concat rows=2" || fail "build_concat" "$OUT"
echo "$OUT" | grep -q "$IDX_CONCAT" && pass "build_concat output_path" || fail "build_concat path" "$OUT"

# ── Test: search concat-built index ──
SQL="
LOAD '$PROJECT_DIR/turbovec.duckdb_extension';
SELECT COUNT(*) FROM turboquant_search('$IDX_CONCAT', '[0.1,0.2,0.3,0.4]', 2);
"
OUT=$($DUCKDB -noheader -c "$SQL" 2>&1)
echo "$OUT" | grep -q "2" && pass "search concat index" || fail "search concat" "$OUT"

echo ""

echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
