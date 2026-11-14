use arrow::{
    array::{Float32Array, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    duckdb_entrypoint_c_api,
    vtab::{arrow::record_batch_to_duckdb_data_chunk, BindInfo, InitInfo, TableFunctionInfo, VTab},
    Connection, Result,
};
use std::{error::Error, sync::Arc};
use rayon::prelude::*;
use turbovec::TurboQuantIndex;

// ── Process-wide index cache ─────────────────────────────────────────────
// Loading an index file is expensive (an 82MB IVF v2 index at 100k vectors
// cost ~34ms per query, 10x the search itself). DuckDB binds table functions
// per query, so without a cache every search re-reads the file. The cache
// keys on the canonical path; entries are Arc'd so concurrent queries share
// one loaded index. TurboQuantIndex::search takes &self with internal
// OnceLock caches, so sharing across threads is safe.
//
// Eviction: simple capacity cap (ENTRY_CAP) on a FIFO basis. moka/LRU would
// add a dependency for little gain at these sizes; the common deployment has
// a handful of hot indexes.
mod idx_cache {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    const ENTRY_CAP: usize = 8;

    struct CacheState {
        map: HashMap<String, Arc<TurboQuantIndex>>,
        order: VecDeque<String>,
    }

    static CACHE: Mutex<Option<CacheState>> = Mutex::new(None);

    pub fn get_or_load(path: &str) -> std::result::Result<Arc<TurboQuantIndex>, String> {
        let canonical = std::fs::canonicalize(path)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string());

        let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let state = guard.get_or_insert_with(|| CacheState {
            map: HashMap::new(),
            order: VecDeque::new(),
        });

        if let Some(hit) = state.map.get(&canonical) {
            return Ok(Arc::clone(hit));
        }
        let idx = TurboQuantIndex::load(&canonical)
            .map_err(|e| format!("index cache load {path}: {e}"))?;
        let idx = Arc::new(idx);
        state.map.insert(canonical.clone(), Arc::clone(&idx));
        state.order.push_back(canonical);
        while state.order.len() > ENTRY_CAP {
            if let Some(evict) = state.order.pop_front() {
                state.map.remove(&evict);
            }
        }
        Ok(idx)
    }
}

// Same cache for IdMapIndex (IVF v2 single-file layout).
mod idmap_cache {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    const ENTRY_CAP: usize = 8;

    struct CacheState {
        map: HashMap<String, Arc<turbovec::IdMapIndex>>,
        order: VecDeque<String>,
    }

    static CACHE: Mutex<Option<CacheState>> = Mutex::new(None);

    pub fn get_or_load(path: &str) -> std::result::Result<Arc<turbovec::IdMapIndex>, String> {
        let canonical = std::fs::canonicalize(path)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string());

        let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let state = guard.get_or_insert_with(|| CacheState {
            map: HashMap::new(),
            order: VecDeque::new(),
        });

        if let Some(hit) = state.map.get(&canonical) {
            return Ok(Arc::clone(hit));
        }
        let idx = turbovec::IdMapIndex::load(&canonical)
            .map_err(|e| format!("idmap cache load {path}: {e}"))?;
        let idx = Arc::new(idx);
        state.map.insert(canonical.clone(), Arc::clone(&idx));
        state.order.push_back(canonical);
        while state.order.len() > ENTRY_CAP {
            if let Some(evict) = state.order.pop_front() {
                state.map.remove(&evict);
            }
        }
        Ok(idx)
    }
}

// ── turboquant_search(index_path, query_str, k) ──

#[repr(C)]
pub struct SearchInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct SearchBindData {
    results: Vec<(i32, f32)>,
    schema: Arc<arrow::datatypes::Schema>,
}

pub struct TurboQuantSearchVTab;

impl VTab for TurboQuantSearchVTab {
    type BindData = SearchBindData;
    type InitData = SearchInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 3 {
            return Err("turboquant_search: expected 3 params (path, query_str, k)".into());
        }
        let path = bind.get_parameter(0).to_string();
        let query = parse_float_array(&bind.get_parameter(1).to_string())?;
        let k: usize = bind.get_parameter(2).to_string().parse()?;
        let idx = idx_cache::get_or_load(&path)
            .map_err(|e| format!("turboquant_search: {e}"))?;
        let n = idx.len();
        let k = k.min(n);  // clamp to index size
        let sr = idx.search(&query, k);

        let mut results = Vec::with_capacity(sr.nq * k);
        for qi in 0..sr.nq {
            let off = qi * k;
            for j in 0..k {
                results.push((sr.indices[off + j] as i32, sr.scores[off + j]));
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));
        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("score", LogicalTypeHandle::from(LogicalTypeId::Float));
        Ok(SearchBindData { results, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(SearchInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();
        let n = bind.results.len();
        if n == 0 {
            init.done.store(true, std::sync::atomic::Ordering::Relaxed);
            output.set_len(0);
            return Ok(());
        }
        let indices: Vec<i32> = bind.results.iter().take(n).map(|r| r.0).collect();
        let scores: Vec<f32> = bind.results.iter().take(n).map(|r| r.1).collect();
        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(indices)),
                Arc::new(Float32Array::from(scores)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ])
    }
}

// ── turboquant_score(index_path, query_str) ──
// Returns TurboQuant quantized similarity scores for ALL vectors.
// NOTE: these are approximate quantized scores, NOT exact cosine.

#[repr(C)]
pub struct ScoreInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct ScoreBindData {
    results: Vec<(i32, f32)>,
    schema: Arc<arrow::datatypes::Schema>,
}

pub struct TurboQuantScoreVTab;

impl VTab for TurboQuantScoreVTab {
    type BindData = ScoreBindData;
    type InitData = ScoreInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 2 {
            return Err("turboquant_score: expected 2 params (path, query_str)".into());
        }
        let path = bind.get_parameter(0).to_string();
        let query = parse_float_array(&bind.get_parameter(1).to_string())?;

        let idx = idx_cache::get_or_load(&path)
            .map_err(|e| format!("turboquant_score: {e}"))?;
        let n = idx.len();
        if n == 0 {
            let schema = Arc::new(Schema::new(vec![
                Field::new("idx", DataType::Int32, false),
                Field::new("score", DataType::Float32, false),
            ]));
            return Ok(ScoreBindData { results: Vec::new(), schema });
        }

        let sr = idx.search(&query, n);
        let results: Vec<(i32, f32)> = (0..n)
            .map(|j| (sr.indices[j] as i32, sr.scores[j]))
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));
        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("score", LogicalTypeHandle::from(LogicalTypeId::Float));
        Ok(ScoreBindData { results, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(ScoreInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();
        let n = bind.results.len();
        if n == 0 {
            init.done.store(true, std::sync::atomic::Ordering::Relaxed);
            output.set_len(0);
            return Ok(());
        }
        let indices: Vec<i32> = bind.results.iter().take(n).map(|r| r.0).collect();
        let scores: Vec<f32> = bind.results.iter().take(n).map(|r| r.1).collect();
        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(indices)),
                Arc::new(Float32Array::from(scores)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── turboquant_build(vectors, dim, bit_width, output_path) ──
// vectors: VARCHAR (nested array string) or LIST<FLOAT> (flat float list)

pub struct TurboQuantBuildVTab;

#[repr(C)]
pub struct BuildInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct BuildBindData {
    output_path: String,
    rows: usize,
}

impl VTab for TurboQuantBuildVTab {
    type BindData = BuildBindData;
    type InitData = BuildInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 4 {
            return Err(
                "turboquant_build: expected 4 params (vectors, dim, bit_width, output_path)"
                    .into(),
            );
        }
        let dim: usize = bind.get_parameter(1).to_string().parse()?;
        let bw: usize = bind.get_parameter(2).to_string().parse()?;
        let output_path = bind.get_parameter(3).to_string();

        let vectors_str = bind.get_parameter(0).to_string();
        let vectors = parse_nested_float_arrays(&vectors_str, dim)?;

        let n = vectors.len() / dim;
        if n == 0 {
            return Err("turboquant_build: no vectors provided".into());
        }

        let mut idx =
            TurboQuantIndex::new(dim, bw).map_err(|e| format!("turboquant_build: {e}"))?;
        idx.add(&vectors);
        idx.write(&output_path)
            .map_err(|e| format!("turboquant_build: write error: {e}"))?;

        bind.add_result_column(
            "output_path",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column("rows", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(BuildBindData {
            output_path,
            rows: n,
        })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(BuildInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_path", DataType::Utf8, false),
            Field::new("rows", DataType::Int32, false),
        ]));
        let path_arr = arrow::array::StringArray::from(vec![bind.output_path.as_str()]);
        let rows_arr = Int32Array::from(vec![bind.rows as i32]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(path_arr), Arc::new(rows_arr)])?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── Utilities ──

fn parse_float_array(raw: &str) -> std::result::Result<Vec<f32>, Box<dyn Error>> {
    let cleaned = raw.trim().trim_start_matches('[').trim_end_matches(']');
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }
    cleaned
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid float '{s}': {e}").into())
        })
        .collect()
}

fn parse_nested_float_arrays(
    raw: &str,
    dim: usize,
) -> std::result::Result<Vec<f32>, Box<dyn Error>> {
    let s = raw.trim();
    let mut result = Vec::new();

    if s.starts_with('[') && !s.starts_with("[[") {
        if let Some(first_close) = s.find(']') {
            let after = s[first_close + 1..].trim();
            if after.starts_with(',') {
                let wrapped = format!("[{}]", s);
                return parse_nested_float_arrays(&wrapped, dim);
            }
        }
        return parse_float_array(s);
    }

    if !s.starts_with("[[") {
        return Err(format!(
            "turboquant_build: expected nested or flat array, got '{}'",
            &s[..64.min(s.len())]
        )
        .into());
    }

    let inner = &s[1..s.len() - 1];
    let mut depth = 0;
    let mut current = String::new();
    for ch in inner.chars() {
        match ch {
            '[' => {
                if depth == 0 {
                    current.clear();
                } else {
                    current.push(ch);
                }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let arr = parse_float_array(&format!("[{}]", current))?;
                    if arr.len() != dim {
                        return Err(format!(
                            "turboquant_build: dim mismatch: expected {dim}, got {}",
                            arr.len()
                        )
                        .into());
                    }
                    result.extend(arr);
                } else {
                    current.push(ch);
                }
            }
            _ => current.push(ch),
        }
    }
    Ok(result)
}

// ── turboquant_build_list(vectors_list, bit_width, output_path) ──
// vectors_list: LIST<FLOAT> — flat float array from
//   (SELECT list(unnested ORDER BY rowid, idx)
//    FROM documents, unnest(emb) WITH ORDINALITY AS t(unnested, idx))
// dim is inferred from the total number of floats (must be a multiple of some dim).

pub struct TurboQuantBuildListVTab;

#[repr(C)]
pub struct BuildListInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct BuildListBindData {
    output_path: String,
    rows: usize,
    dim: usize,
}

impl VTab for TurboQuantBuildListVTab {
    type BindData = BuildListBindData;
    type InitData = BuildListInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 3 {
            return Err(
                "turboquant_build_list: expected 3 params (vectors_list, bit_width, output_path)"
                    .into(),
            );
        }

        // Parse vectors from LIST<FLOAT>
        let list_val = bind.get_parameter(0);
        let floats = extract_list_floats(&list_val)
            .map_err(|e| format!("turboquant_build_list: {e}"))?;
        if floats.is_empty() {
            return Err("turboquant_build_list: empty vectors list".into());
        }

        let bw: usize = bind.get_parameter(1).to_string().parse()?;
        let output_path = bind.get_parameter(2).to_string();

        // Auto-detect dim from the first few common dims that evenly divide total
        let dim = auto_detect_dim(floats.len())
            .ok_or_else(|| {
                format!(
                    "turboquant_build_list: cannot auto-detect dim from {} values (must be multiple of 8, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)",
                    floats.len()
                )
            })?;

        if floats.len() % dim != 0 {
            return Err(format!(
                "turboquant_build_list: {} floats not evenly divisible by dim {}",
                floats.len(),
                dim
            )
            .into());
        }

        let n = floats.len() / dim;
        let mut idx =
            TurboQuantIndex::new(dim, bw).map_err(|e| format!("turboquant_build_list: {e}"))?;
        idx.add(&floats);
        idx.write(&output_path)
            .map_err(|e| format!("turboquant_build_list: write error: {e}"))?;

        bind.add_result_column(
            "output_path",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column("rows", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("dim", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(BuildListBindData {
            output_path,
            rows: n,
            dim,
        })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(BuildListInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_path", DataType::Utf8, false),
            Field::new("rows", DataType::Int32, false),
            Field::new("dim", DataType::Int32, false),
        ]));
        let path_arr = arrow::array::StringArray::from(vec![bind.output_path.as_str()]);
        let rows_arr = Int32Array::from(vec![bind.rows as i32]);
        let dim_arr = Int32Array::from(vec![bind.dim as i32]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(path_arr), Arc::new(rows_arr), Arc::new(dim_arr)],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let float_list =
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float));
        Some(vec![
            float_list,
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

/// Extract flat Vec<f32> from a DuckDB LIST<FLOAT>
fn extract_list_floats(
    val: &duckdb::vtab::Value,
) -> std::result::Result<Vec<f32>, Box<dyn Error>> {
    let items = val
        .to_list()
        .ok_or_else(|| "expected LIST<FLOAT> value".to_string())?;
    let mut floats = Vec::with_capacity(items.len());
    for item in &items {
        floats.push(item.to_float());
    }
    Ok(floats)
}

/// Auto-detect vector dimension from total float count
fn auto_detect_dim(total: usize) -> Option<usize> {
    // Common embedding dimensions, tried in order
    const CANDIDATES: &[usize] = &[
        1536, 1024, 768, 512, 384, 256, 128, 64, 8, 2048, 3072, 4096, 8192,
    ];
    CANDIDATES
        .iter()
        .find(|&&d| total % d == 0 && total >= d)
        .copied()
}

// ── IVF Index ──

/// Simple IVF index: K-means partitions → one TurboQuantIndex per cluster
struct IvfMeta {
    dim: usize,
    bit_width: usize,
    num_lists: usize,
    centroids: Vec<f32>, // flat: num_lists * dim
}

impl IvfMeta {
    fn write(&self, path: &str) -> std::result::Result<(), Box<dyn Error>> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&self.dim.to_le_bytes())?;
        f.write_all(&self.bit_width.to_le_bytes())?;
        f.write_all(&self.num_lists.to_le_bytes())?;
        for c in &self.centroids {
            f.write_all(&c.to_le_bytes())?;
        }
        Ok(())
    }

    fn load(path: &str) -> std::result::Result<Self, Box<dyn Error>> {
        let data = std::fs::read(path)?;
        let dim = usize::from_le_bytes(data[0..8].try_into().unwrap());
        let bit_width = usize::from_le_bytes(data[8..16].try_into().unwrap());
        let num_lists = usize::from_le_bytes(data[16..24].try_into().unwrap());
        let centroids: Vec<f32> = data[24..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if centroids.len() != num_lists * dim {
            return Err("corrupt IVF meta: centroid count mismatch".into());
        }
        Ok(Self { dim, bit_width, num_lists, centroids })
    }
}

/// IVF v2 meta: single IdMapIndex (`index.tv`) with vectors stored in
/// CLUSTER-CONTIGUOUS slot order — cluster ci occupies slots
/// [cluster_offsets[ci], cluster_offsets[ci+1]). One rotation matrix and
/// codebook for the whole IVF index (the v1 per-cluster-files layout paid
/// ~1.6 MB of fixed overhead per cluster: 32 clusters on 1.5k docs = 19×
/// the flat index size; 316 clusters on 100k = 0.6 GB vs 77 MB flat).
struct IvfMetaV2 {
    dim: usize,
    bit_width: usize,
    num_lists: usize,
    centroids: Vec<f32>,       // flat: num_lists * dim
    cluster_offsets: Vec<u32>, // num_lists + 1; slot ranges per cluster
    slot_to_id: Vec<i32>,      // slot → original row id
}

impl IvfMetaV2 {
    fn write(&self, path: &str) -> std::result::Result<(), Box<dyn Error>> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"TIVF2\0")?;
        f.write_all(&(self.dim as u64).to_le_bytes())?;
        f.write_all(&(self.bit_width as u64).to_le_bytes())?;
        f.write_all(&(self.num_lists as u64).to_le_bytes())?;
        f.write_all(&(self.cluster_offsets.len() as u64).to_le_bytes())?;
        f.write_all(&(self.slot_to_id.len() as u64).to_le_bytes())?;
        for c in &self.centroids {
            f.write_all(&c.to_le_bytes())?;
        }
        for o in &self.cluster_offsets {
            f.write_all(&o.to_le_bytes())?;
        }
        for id in &self.slot_to_id {
            f.write_all(&id.to_le_bytes())?;
        }
        Ok(())
    }

    fn load(path: &str) -> std::result::Result<Self, Box<dyn Error>> {
        let data = std::fs::read(path)?;
        if data.len() < 6 + 5 * 8 || &data[0..6] != b"TIVF2\0" {
            return Err("not an IVF v2 meta file (missing TIVF2 magic)".into());
        }
        let mut off = 6;
        let rd_u64 = |off: &mut usize| -> u64 {
            let v = u64::from_le_bytes(data[*off..*off + 8].try_into().unwrap());
            *off += 8;
            v
        };
        let dim = rd_u64(&mut off) as usize;
        let bit_width = rd_u64(&mut off) as usize;
        let num_lists = rd_u64(&mut off) as usize;
        let n_off = rd_u64(&mut off) as usize;
        let n_ids = rd_u64(&mut off) as usize;
        let centroids: Vec<f32> = data[off..off + num_lists * dim * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        off += num_lists * dim * 4;
        let cluster_offsets: Vec<u32> = data[off..off + n_off * 4]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        off += n_off * 4;
        let slot_to_id: Vec<i32> = data[off..off + n_ids * 4]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if centroids.len() != num_lists * dim || cluster_offsets.len() != num_lists + 1 {
            return Err("corrupt IVF v2 meta: section size mismatch".into());
        }
        Ok(Self { dim, bit_width, num_lists, centroids, cluster_offsets, slot_to_id })
    }
}

/// K-means++ seeding: pick the first centroid at random, then each next
/// centroid with probability proportional to the squared distance to the
/// nearest already-picked centroid. Far better cluster separation than the
/// old "first k evenly-spaced vectors" init, which could seed near-duplicate
/// centroids and produce empty/skewed clusters (lower IVF recall).
///
/// Seeding runs on a random SAMPLE (`sample_cap`) when n is large: the
/// full-population roulette is O(n·k·dim), which at 100k×316×1536 adds
/// minutes for no measurable quality gain over a 10k sample.
fn kmeans_pp_init(vectors: &[f32], dim: usize, k: usize, rng: &mut impl FnMut() -> usize) -> Vec<f32> {
    const SAMPLE_CAP: usize = 10_000;
    let n = vectors.len() / dim;
    let sample: Vec<usize> = if n <= SAMPLE_CAP {
        (0..n).collect()
    } else {
        // deterministic stride sample (reproducible builds)
        let stride = n / SAMPLE_CAP;
        (0..SAMPLE_CAP).map(|i| i * stride).collect()
    };
    let sn = sample.len();

    let mut centroids = vec![0.0f32; k * dim];
    let first = sample[rng() % sn];
    centroids[..dim].copy_from_slice(&vectors[first * dim..(first + 1) * dim]);

    // dist[i] = squared L2 from SAMPLE vector i to its nearest chosen centroid
    let mut dist: Vec<f32> = (0..sn)
        .map(|i| {
            let v = &vectors[sample[i] * dim..(sample[i] + 1) * dim];
            let c = &centroids[0..dim];
            let d = l2_dist(v, c);
            d * d
        })
        .collect();

    for ci in 1..k {
        let total: f32 = dist.iter().sum();
        let mut pick = if total > 0.0 { (rng() as f32 / usize::MAX as f32) * total } else { 0.0 };
        let mut chosen = sn - 1;
        for (i, &d) in dist.iter().enumerate() {
            pick -= d;
            if pick <= 0.0 {
                chosen = i;
                break;
            }
        }
        let vi = sample[chosen];
        centroids[ci * dim..(ci + 1) * dim].copy_from_slice(&vectors[vi * dim..(vi + 1) * dim]);
        let c = &centroids[ci * dim..(ci + 1) * dim];
        for i in 0..sn {
            let v = &vectors[sample[i] * dim..(sample[i] + 1) * dim];
            let nd = l2_dist(v, c);
            let nd2 = nd * nd;
            if nd2 < dist[i] {
                dist[i] = nd2;
            }
        }
    }
    centroids
}

/// K-means clustering (Lloyd's algorithm) with k-means++ seeding
fn kmeans(
    vectors: &[f32], dim: usize, k: usize, max_iters: usize,
) -> (Vec<f32>, Vec<usize>) {
    let n = vectors.len() / dim;
    let mut centroids = vec![0.0f32; k * dim];
    let mut assignments = vec![0usize; n];

    // Seed with k-means++ (deterministic PRNG so builds are reproducible)
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as usize
    };
    centroids = kmeans_pp_init(vectors, dim, k, &mut rng);

    for _iter in 0..max_iters {
        // Assign each vector to nearest centroid (parallel across vectors;
        // this O(n·k·dim) loop dominates build time at scale)
        let new_assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|vi| {
                let v = &vectors[vi * dim..(vi + 1) * dim];
                let mut best_c = 0usize;
                let mut best_d = l2_dist(v, &centroids[0..dim]);
                for ci in 1..k {
                    let d = l2_dist(v, &centroids[ci * dim..(ci + 1) * dim]);
                    if d < best_d {
                        best_d = d;
                        best_c = ci;
                    }
                }
                best_c
            })
            .collect();
        let changed_any = new_assignments
            .iter()
            .zip(assignments.iter())
            .filter(|(a, b)| a != b)
            .count();
        assignments.copy_from_slice(&new_assignments);
        if changed_any == 0 { break; }

        // Recompute centroids
        let mut counts = vec![0usize; k];
        centroids.fill(0.0);
        for vi in 0..n {
            let c = assignments[vi];
            counts[c] += 1;
            for d in 0..dim {
                centroids[c * dim + d] += vectors[vi * dim + d];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for d in 0..dim {
                    centroids[ci * dim + d] /= counts[ci] as f32;
                }
            }
        }
    }

    (centroids, assignments)
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

/// Load per-cluster idmap: Vec<local_pos → global_idx>
fn load_idmap(path: &str) -> Vec<i32> {
    std::fs::read(path)
        .map(|data| {
            data.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .unwrap_or_default()
}


// ── turboquant_build_ivf
pub struct IvfBuildVTab;

#[repr(C)]
pub struct IvfBuildInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct IvfBuildBindData {
    output_dir: String,
    rows: usize,
    num_lists: usize,
}

impl VTab for IvfBuildVTab {
    type BindData = IvfBuildBindData;
    type InitData = IvfBuildInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 5 {
            return Err(
                "turboquant_build_ivf: expected 5 params (vectors_str, dim, bit_width, num_lists, output_dir)"
                    .into(),
            );
        }
        let vectors_str = bind.get_parameter(0).to_string();
        let dim: usize = bind.get_parameter(1).to_string().parse()?;
        let bw: usize = bind.get_parameter(2).to_string().parse()?;
        let output_dir = bind.get_parameter(4).to_string();

        let vectors = parse_nested_float_arrays(&vectors_str, dim)?;
        let n = vectors.len() / dim;
        if n == 0 {
            return Err("turboquant_build_ivf: no vectors provided".into());
        }
        // num_lists = 0 → auto: sqrt(n) heuristic (DuckIR-style default; keeps
        // per-cluster index count from exploding on small datasets where each
        // .tv file carries ~1.6 MB of fixed overhead: rotation matrix +
        // codebook. 32 clusters on 1.5k docs = 19× the flat index size).
        let num_lists_param: i64 = bind.get_parameter(3).to_string().parse()?;
        let lists = if num_lists_param == 0 {
            ((n as f64).sqrt().round() as usize).max(1)
        } else {
            (num_lists_param as usize).min(n)
        };
        if lists < 1 {
            return Err("turboquant_build_ivf: num_lists must be >= 0 (0 = auto sqrt(n))".into());
        }

        // K-means clustering
        let (centroids, assignments) = kmeans(&vectors, dim, lists, 20);

        // V2 layout: count cluster sizes, compute slot offsets, then gather
        // vectors cluster-contiguously into ONE IdMapIndex (id = original
        // row) — a single rotation matrix/codebook instead of `lists` copies.
        let mut counts = vec![0u32; lists];
        for &c in &assignments {
            counts[c] += 1;
        }
        let mut cluster_offsets = Vec::with_capacity(lists + 1);
        let mut acc = 0u32;
        cluster_offsets.push(0);
        for &c in &counts {
            acc += c;
            cluster_offsets.push(acc);
        }
        // next_slot[ci] = write cursor within cluster ci's slot range
        let mut next_slot = cluster_offsets[..lists].to_vec();
        let mut gathered = vec![0.0f32; n * dim];
        let mut slot_to_id = vec![0i32; n];
        for vi in 0..n {
            let c = assignments[vi];
            let slot = next_slot[c] as usize;
            next_slot[c] += 1;
            gathered[slot * dim..(slot + 1) * dim]
                .copy_from_slice(&vectors[vi * dim..(vi + 1) * dim]);
            slot_to_id[slot] = vi as i32;
        }

        std::fs::create_dir_all(&output_dir)?;
        // IdMapIndex wants u64 ids; slot_to_id is the id source, and since
        // ids here are dense row numbers we can use slot==id mapping directly
        // via ids = gathered order (id = slot_to_id[slot]).
        let ids: Vec<u64> = slot_to_id.iter().map(|&v| v as u64).collect();
        let mut idmap_index = turbovec::IdMapIndex::new(dim, bw)
            .map_err(|e| format!("turboquant_build_ivf: idmap index: {e}"))?;
        idmap_index
            .add_with_ids(&gathered, &ids)
            .map_err(|e| format!("turboquant_build_ivf: add_with_ids: {e}"))?;
        idmap_index
            .write(&format!("{output_dir}/index.tv"))
            .map_err(|e| format!("turboquant_build_ivf: write index: {e}"))?;

        // Write v2 meta (also covers empty clusters: zero-width ranges)
        let meta = IvfMetaV2 {
            dim,
            bit_width: bw,
            num_lists: lists,
            centroids,
            cluster_offsets,
            slot_to_id,
        };
        meta.write(&format!("{output_dir}/meta2.bin"))?;

        bind.add_result_column("output_dir", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("rows", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("num_lists", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(IvfBuildBindData { output_dir, rows: n, num_lists: lists })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(IvfBuildInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_dir", DataType::Utf8, false),
            Field::new("rows", DataType::Int32, false),
            Field::new("num_lists", DataType::Int32, false),
        ]));
        let dir_arr = arrow::array::StringArray::from(vec![bind.output_dir.as_str()]);
        let rows_arr = Int32Array::from(vec![bind.rows as i32]);
        let lists_arr = Int32Array::from(vec![bind.num_lists as i32]);
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(dir_arr), Arc::new(rows_arr), Arc::new(lists_arr),
        ])?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── turboquant_search_ivf(index_dir, query_str, k, probes) ──

pub struct IvfSearchVTab;

#[repr(C)]
pub struct IvfSearchInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct IvfSearchBindData {
    results: Vec<(i32, f32)>,
    schema: Arc<arrow::datatypes::Schema>,
}

impl VTab for IvfSearchVTab {
    type BindData = IvfSearchBindData;
    type InitData = IvfSearchInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 4 {
            return Err(
                "turboquant_search_ivf: expected 4 params (index_dir, query_str, k, probes)"
                    .into(),
            );
        }
        let dir = bind.get_parameter(0).to_string();
        let query = parse_float_array(&bind.get_parameter(1).to_string())?;
        let k: usize = bind.get_parameter(2).to_string().parse()?;
        let probes: usize = bind.get_parameter(3).to_string().parse()?;

        // V2 single-index layout takes priority; fall back to v1 per-cluster
        // files for indexes built before this version.
        let meta2_path = format!("{dir}/meta2.bin");
        let mut all: Vec<(i32, f32)> = if std::path::Path::new(&meta2_path).exists() {
            let meta = IvfMetaV2::load(&meta2_path)?;
            let dim = meta.dim;
            let probes = probes.min(meta.num_lists);

            // rank centroids by L2 to the query
            let mut centroid_dists: Vec<(usize, f32)> = (0..meta.num_lists)
                .map(|ci| {
                    let c = &meta.centroids[ci * dim..(ci + 1) * dim];
                    (ci, l2_dist(&query, c))
                })
                .collect();
            centroid_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

            // allowlist = external ids in the probed clusters' slot ranges
            let mut allowed: Vec<u64> = Vec::new();
            for &(ci, _) in centroid_dists.iter().take(probes) {
                let (s, e) = (meta.cluster_offsets[ci] as usize, meta.cluster_offsets[ci + 1] as usize);
                for slot in s..e {
                    allowed.push(meta.slot_to_id[slot] as u64);
                }
            }
            if allowed.is_empty() {
                return Ok(IvfSearchBindData { results: Vec::new(), schema: Arc::new(Schema::new(vec![
                    Field::new("idx", DataType::Int32, false),
                    Field::new("score", DataType::Float32, false),
                ])) });
            }

            let idx = idmap_cache::get_or_load(&format!("{dir}/index.tv"))
                .map_err(|e| format!("turboquant_search_ivf: {e}"))?;
            let (scores, ids) = idx
                .search_with_allowlist(&query, k, Some(&allowed))
                .map_err(|e| format!("turboquant_search_ivf: {e}"))?;
            ids.iter().zip(scores.iter())
                .map(|(&id, &s)| (id as i32, s))
                .collect()
        } else {
            // legacy v1 path
            let meta = IvfMeta::load(&format!("{dir}/meta.bin"))?;
            let dim = meta.dim;
            let probes = if probes == 0 || probes == usize::MAX {
                meta.num_lists
            } else {
                probes.min(meta.num_lists)
            };

            let mut centroid_dists: Vec<(usize, f32)> = (0..meta.num_lists)
                .map(|ci| {
                    let c = &meta.centroids[ci * dim..(ci + 1) * dim];
                    (ci, l2_dist(&query, c))
                })
                .collect();
            centroid_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

            let mut all: Vec<(i32, f32)> = Vec::new();
            for (ci, _) in centroid_dists.iter().take(probes) {
                let path = format!("{dir}/cluster_{ci}.tv");
                if let Ok(idx) = TurboQuantIndex::load(&path) {
                    let n = idx.len();
                    if n == 0 { continue; }
                    let idmap = load_idmap(&format!("{dir}/idmap_{ci}.bin"));
                    let sr = idx.search(&query, k.min(n));
                    for j in 0..k.min(n) {
                        let local = sr.indices[j] as usize;
                        let global = if local < idmap.len() { idmap[local] } else { local as i32 };
                        all.push((global, sr.scores[j]));
                    }
                }
            }
            all.sort_by(|a, b| b.1.total_cmp(&a.1));
            all.truncate(k);
            all
        };

        // Sort and take top-k (total_cmp: NaN-safe, see note above)
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        all.truncate(k);

        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));
        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("score", LogicalTypeHandle::from(LogicalTypeId::Float));
        Ok(IvfSearchBindData { results: all, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(IvfSearchInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();
        let n = bind.results.len();
        if n == 0 {
            init.done.store(true, std::sync::atomic::Ordering::Relaxed);
            output.set_len(0);
            return Ok(());
        }
        let indices: Vec<i32> = bind.results.iter().take(n).map(|r| r.0).collect();
        let scores: Vec<f32> = bind.results.iter().take(n).map(|r| r.1).collect();
        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(indices)),
                Arc::new(Float32Array::from(scores)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ])
    }
}

// ── turboquant_add(index_path, vectors_str, dim) ──
// Load existing index, append vectors, write back.

pub struct TurboQuantAddVTab;

#[repr(C)]
pub struct AddInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct AddBindData {
    output_path: String,
    added: usize,
    total: usize,
}

impl VTab for TurboQuantAddVTab {
    type BindData = AddBindData;
    type InitData = AddInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 3 {
            return Err(
                "turboquant_add: expected 3 params (path, vectors_str, dim)".into(),
            );
        }
        let path = bind.get_parameter(0).to_string();
        let vectors_str = bind.get_parameter(1).to_string();
        let dim: usize = bind.get_parameter(2).to_string().parse()?;

        let vectors = parse_nested_float_arrays(&vectors_str, dim)?;
        let to_add = vectors.len() / dim;
        if to_add == 0 {
            return Err("turboquant_add: no vectors provided".into());
        }

        let mut idx = TurboQuantIndex::load(&path)
            .map_err(|e| format!("turboquant_add: load error: {e}"))?;
        let before = idx.len();
        idx.add(&vectors);
        idx.write(&path)
            .map_err(|e| format!("turboquant_add: write error: {e}"))?;

        bind.add_result_column("output_path", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("added", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("total", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(AddBindData { output_path: path, added: to_add, total: before + to_add })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(AddInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0); return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_path", DataType::Utf8, false),
            Field::new("added", DataType::Int32, false),
            Field::new("total", DataType::Int32, false),
        ]));
        let path_arr = arrow::array::StringArray::from(vec![bind.output_path.as_str()]);
        let added_arr = Int32Array::from(vec![bind.added as i32]);
        let total_arr = Int32Array::from(vec![bind.total as i32]);
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(path_arr), Arc::new(added_arr), Arc::new(total_arr),
        ])?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ])
    }
}

// ── turboquant_remove(index_path, idx) ──
// Remove a vector by index with swap_remove (O(1)), rewrite file.

pub struct TurboQuantRemoveVTab;

#[repr(C)]
pub struct RemoveInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct RemoveBindData {
    output_path: String,
    removed_idx: usize,
    remaining: usize,
}

impl VTab for TurboQuantRemoveVTab {
    type BindData = RemoveBindData;
    type InitData = RemoveInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 2 {
            return Err("turboquant_remove: expected 2 params (path, idx)".into());
        }
        let path = bind.get_parameter(0).to_string();
        let target: usize = bind.get_parameter(1).to_string().parse()?;

        let mut idx = TurboQuantIndex::load(&path)
            .map_err(|e| format!("turboquant_remove: load error: {e}"))?;
        let before = idx.len();
        if target >= before {
            return Err(format!(
                "turboquant_remove: idx {target} out of range (0..{before})"
            ).into());
        }
        idx.swap_remove(target);
        idx.write(&path)
            .map_err(|e| format!("turboquant_remove: write error: {e}"))?;

        bind.add_result_column("output_path", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("removed_idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("remaining", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(RemoveBindData { output_path: path, removed_idx: target, remaining: before - 1 })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(RemoveInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0); return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_path", DataType::Utf8, false),
            Field::new("removed_idx", DataType::Int32, false),
            Field::new("remaining", DataType::Int32, false),
        ]));
        let path_arr = arrow::array::StringArray::from(vec![bind.output_path.as_str()]);
        let ridx_arr = Int32Array::from(vec![bind.removed_idx as i32]);
        let rem_arr = Int32Array::from(vec![bind.remaining as i32]);
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(path_arr), Arc::new(ridx_arr), Arc::new(rem_arr),
        ])?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ])
    }
}

// ── turboquant_build_concat(output_path, dim, bit_width, values_str) ──
// Build from flat comma-separated float list (output of string_agg).
// Avoids nested-array string format. dim passed explicitly.
// Usage: SELECT * FROM turboquant_build_concat('/tmp/idx.tv', 1024, 4, getvariable('v'));

pub struct TurboQuantBuildConcatVTab;

#[repr(C)]
pub struct BuildConcatInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct BuildConcatBindData {
    output_path: String,
    rows: usize,
}

impl VTab for TurboQuantBuildConcatVTab {
    type BindData = BuildConcatBindData;
    type InitData = BuildConcatInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 4 {
            return Err(
                "turboquant_build_concat: expected 4 params (output_path, dim, bit_width, values_str)"
                    .into(),
            );
        }
        let output_path = bind.get_parameter(0).to_string();
        let dim: usize = bind.get_parameter(1).to_string().parse()?;
        let bw: usize = bind.get_parameter(2).to_string().parse()?;
        let values_str = bind.get_parameter(3).to_string();

        // Parse flat comma-separated floats (NOT nested arrays)
        let all = parse_float_array(&format!("[{}]", values_str))?;
        if all.len() % dim != 0 {
            return Err(format!(
                "turboquant_build_concat: {} floats not divisible by dim {}", all.len(), dim
            ).into());
        }
        let n = all.len() / dim;
        if n == 0 {
            return Err("turboquant_build_concat: no vectors".into());
        }

        let mut idx = TurboQuantIndex::new(dim, bw)
            .map_err(|e| format!("turboquant_build_concat: {e}"))?;
        idx.add(&all);
        idx.write(&output_path)
            .map_err(|e| format!("turboquant_build_concat: write error: {e}"))?;

        bind.add_result_column("output_path", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("rows", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(BuildConcatBindData { output_path, rows: n })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(BuildConcatInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0); return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_path", DataType::Utf8, false),
            Field::new("rows", DataType::Int32, false),
        ]));
        let path_arr = arrow::array::StringArray::from(vec![bind.output_path.as_str()]);
        let rows_arr = Int32Array::from(vec![bind.rows as i32]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(path_arr), Arc::new(rows_arr)])?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── turboquant_search_within(index_path, query_str, eps, k) ──
// Radius (WITHIN) counterpart to turboquant_search: instead of "the k
// nearest", return every vector whose (approximate quantized) similarity
// score is within `eps` of the best (nearest) result — i.e. everything as
// close to the query as the nearest neighbor, within tolerance eps.
// `k` caps the output (0 = no cap); passing k>0 fuses top-k with the radius
// bound (returns the k nearest that are within eps).
//
// NOTE: scores are approximate quantized similarity (higher = nearer,
// best-first), so `eps` is a score gap in the index's own units and the
// bound is approximate (like pgvector's HNSW radius). Calibrate eps
// empirically per index; it is not comparable across engines.

#[repr(C)]
pub struct WithinInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct WithinBindData {
    results: Vec<(i32, f32)>,
    schema: Arc<arrow::datatypes::Schema>,
}

pub struct TurboQuantSearchWithinVTab;

impl VTab for TurboQuantSearchWithinVTab {
    type BindData = WithinBindData;
    type InitData = WithinInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 4 {
            return Err(
                "turboquant_search_within: expected 4 params (path, query_str, eps, k)".into(),
            );
        }
        let path = bind.get_parameter(0).to_string();
        let query = parse_float_array(&bind.get_parameter(1).to_string())?;
        let eps: f32 = bind
            .get_parameter(2)
            .to_string()
            .parse()
            .map_err(|_| "turboquant_search_within: eps must be a number")?;
        let k: usize = bind
            .get_parameter(3)
            .to_string()
            .parse()
            .map_err(|_| "turboquant_search_within: k must be a non-negative integer")?;

        let idx = TurboQuantIndex::load(&path)
            .map_err(|e| format!("turboquant_search_within: {e}"))?;
        let n = idx.len();

        let mut results = Vec::new();
        if n > 0 {
            // Full scan. TurboQuant's search() is not guaranteed to emit
            // results in strictly score-descending order, so we cannot
            // early-break on the first below-threshold row. Instead collect
            // every vector whose score is within eps of the best, then sort
            // by score (best-first) and apply the k cap — giving "the k
            // nearest that are within eps" (top-k + radius fused).
            let sr = idx.search(&query, n);
            let best = sr.scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let threshold = best - eps;
            for j in 0..sr.indices.len() {
                if sr.scores[j] >= threshold {
                    results.push((sr.indices[j] as i32, sr.scores[j]));
                }
            }
            results.sort_by(|a, b| b.1.total_cmp(&a.1));
            if k > 0 {
                results.truncate(k);
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));
        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("score", LogicalTypeHandle::from(LogicalTypeId::Float));
        Ok(WithinBindData { results, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(WithinInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();
        let n = bind.results.len();
        if n == 0 {
            init.done.store(true, std::sync::atomic::Ordering::Relaxed);
            output.set_len(0);
            return Ok(());
        }
        let indices: Vec<i32> = bind.results.iter().take(n).map(|r| r.0).collect();
        let scores: Vec<f32> = bind.results.iter().take(n).map(|r| r.1).collect();
        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(indices)),
                Arc::new(Float32Array::from(scores)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Double),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ])
    }
}

// ── Entrypoint ──

#[duckdb_entrypoint_c_api(ext_name = "turbovec")]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<TurboQuantSearchVTab>("turboquant_search")?;
    con.register_table_function::<TurboQuantSearchWithinVTab>("turboquant_search_within")?;
    con.register_table_function::<TurboQuantScoreVTab>("turboquant_score")?;
    con.register_table_function::<TurboQuantBuildVTab>("turboquant_build")?;
    con.register_table_function::<TurboQuantBuildListVTab>("turboquant_build_list")?;
    con.register_table_function::<TurboQuantBuildConcatVTab>("turboquant_build_concat")?;
    con.register_table_function::<TurboQuantAddVTab>("turboquant_add")?;
    con.register_table_function::<TurboQuantRemoveVTab>("turboquant_remove")?;
    con.register_table_function::<IvfBuildVTab>("turboquant_build_ivf")?;
    con.register_table_function::<IvfSearchVTab>("turboquant_search_ivf")?;
    con.register_table_function::<EnsembleBuildVTab>("turboquant_build_ensemble")?;
    con.register_table_function::<EnsembleSearchVTab>("turboquant_search_ensemble")?;
    Ok(())
}

// ── turboquant_build_ensemble(vectors_str, dim, bit_width, members, output_dir) ──
// ── turboquant_search_ensemble(ensemble_dir, query_str, k) ──
//
// Recall recovery via ensembling: the kernel hard-caps bit_width at 2..=4,
// so there is no native 8-bit. Instead, build `members` indexes over the
// SAME vectors, each behind its own deterministic dimension permutation
// (input-side decorrelation: the kernel's internal rotation seed is fixed
// at 42, so permuting the input is the only way to de-correlate quantization
// noise across members). Search runs every member and AVERAGES the scores
// per vector id. Independent noise averages down ~1/sqrt(m): with 2 members
// at 4-bit each, memory is the same as a hypothetical 8-bit index while
// recall lands between 4-bit and exact.

/// Deterministic seeded permutation (splitmix-style shuffle) of 0..dim.
/// Same member index → same permutation on every build, so a rebuilt
/// ensemble stays compatible with its meta.
fn seeded_permutation(dim: usize, member: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..dim).collect();
    let mut s = seed ^ (member as u64).wrapping_mul(0x9E3779B97F4A7C15);
    for i in (1..dim).rev() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let j = (s as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm
}

pub struct EnsembleBuildVTab;

#[repr(C)]
pub struct EnsembleBuildInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct EnsembleBuildBindData {
    output_dir: String,
    rows: usize,
    members: usize,
}

impl VTab for EnsembleBuildVTab {
    type BindData = EnsembleBuildBindData;
    type InitData = EnsembleBuildInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let vectors_str = bind.get_parameter(0).to_string();
        let dim: usize = bind.get_parameter(1).to_string().parse()?;
        let bw: usize = bind.get_parameter(2).to_string().parse()?;
        let members: usize = bind.get_parameter(3).to_string().parse()?;
        let output_dir = bind.get_parameter(4).to_string();

        if !(2..=4).contains(&bw) {
            return Err("turboquant_build_ensemble: bit_width must be 2, 3 or 4".into());
        }
        if members < 1 || members > 8 {
            return Err("turboquant_build_ensemble: members must be 1..=8".into());
        }

        let vectors = parse_nested_float_arrays(&vectors_str, dim)?;
        let n = vectors.len() / dim;
        if n == 0 {
            return Err("turboquant_build_ensemble: no vectors provided".into());
        }

        std::fs::create_dir_all(&output_dir)?;

        // Per member: permute dims, build, write, record permutation.
        // Permutations live in meta so search reproduces them exactly.
        let mut perms: Vec<Vec<u32>> = Vec::with_capacity(members);
        for m in 0..members {
            let perm = seeded_permutation(dim, m, 0x524F54_41525A21); // "ROTARZ!"
            let mut gathered = vec![0.0f32; n * dim];
            for vi in 0..n {
                for d in 0..dim {
                    gathered[vi * dim + d] = vectors[vi * dim + perm[d]];
                }
            }
            let mut idx = TurboQuantIndex::new(dim, bw)
                .map_err(|e| format!("turboquant_build_ensemble member {m}: {e}"))?;
            idx.add(&gathered);
            idx.write(&format!("{output_dir}/member_{m}.tv"))
                .map_err(|e| format!("turboquant_build_ensemble write {m}: {e}"))?;
            perms.push(perm.iter().map(|&p| p as u32).collect());
        }

        // meta: magic, dim, bw, members, n, then each permutation
        use std::io::Write;
        let mut f = std::fs::File::create(format!("{output_dir}/ensemble.bin"))?;
        f.write_all(b"TENS\0")?;
        f.write_all(&(dim as u64).to_le_bytes())?;
        f.write_all(&(bw as u64).to_le_bytes())?;
        f.write_all(&(members as u64).to_le_bytes())?;
        f.write_all(&(n as u64).to_le_bytes())?;
        for perm in &perms {
            for &p in perm {
                f.write_all(&p.to_le_bytes())?;
            }
        }

        bind.add_result_column("output_dir", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("rows", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("members", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(EnsembleBuildBindData { output_dir, rows: n, members })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(EnsembleBuildInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let schema = Arc::new(Schema::new(vec![
            Field::new("output_dir", DataType::Utf8, false),
            Field::new("rows", DataType::Int32, false),
            Field::new("members", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(arrow::array::StringArray::from(vec![bind.output_dir.as_str()])),
            Arc::new(Int32Array::from(vec![bind.rows as i32])),
            Arc::new(Int32Array::from(vec![bind.members as i32])),
        ])?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

pub struct EnsembleSearchVTab;

#[repr(C)]
pub struct EnsembleSearchInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
pub struct EnsembleSearchBindData {
    results: Vec<(i32, f32)>,
    schema: Arc<arrow::datatypes::Schema>,
}

impl VTab for EnsembleSearchVTab {
    type BindData = EnsembleSearchBindData;
    type InitData = EnsembleSearchInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let dir = bind.get_parameter(0).to_string();
        let query = parse_float_array(&bind.get_parameter(1).to_string())?;
        let k: usize = bind.get_parameter(2).to_string().parse()?;

        // load meta
        let data = std::fs::read(format!("{dir}/ensemble.bin"))?;
        if &data[0..5] != b"TENS\0" {
            return Err("not an ensemble index (missing TENS magic)".into());
        }
        let dim = u64::from_le_bytes(data[5..13].try_into().unwrap()) as usize;
        let members = u64::from_le_bytes(data[21..29].try_into().unwrap()) as usize;
        let mut off = 29 + 8; // skip n
        let perms: Vec<Vec<u32>> = (0..members)
            .map(|m| {
                let p: Vec<u32> = data[off + m * dim * 4..off + (m + 1) * dim * 4]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                p
            })
            .collect();

        // sum scores per id across members (ids are positional row numbers)
        use std::collections::HashMap;
        let mut score_sum: HashMap<i32, (f32, u32)> = HashMap::new();
        for m in 0..members {
            // apply this member's permutation to the query
            let perm = &perms[m];
            let mut pq = vec![0.0f32; dim];
            for d in 0..dim {
                pq[d] = query[perm[d] as usize];
            }
            let idx = idx_cache::get_or_load(&format!("{dir}/member_{m}.tv"))?;
            // Ask each member for MORE than k (k per member would truncate
            // the union); fetch min(len, 4k) and let the fused ranking decide.
            let want = (k * 4).min(idx.len()).max(k);
            let sr = idx.search(&pq, want);
            // SearchResults: scores/indices row-major; single query → one row
            let n_res = sr.scores.len();
            for j in 0..n_res {
                let id = sr.indices[j];
                let s = sr.scores[j];
                let e = score_sum.entry(id as i32).or_insert((0.0, 0));
                e.0 += s;
                e.1 += 1;
            }
        }

        let mut results: Vec<(i32, f32)> = score_sum
            .into_iter()
            .map(|(id, (s, cnt))| (id, s / cnt as f32))
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);

        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("score", LogicalTypeHandle::from(LogicalTypeId::Float));

        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));
        Ok(EnsembleSearchBindData { results, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(EnsembleSearchInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        let bind = func.get_bind_data();
        let n = bind.results.len();
        if n == 0 {
            output.set_len(0);
            return Ok(());
        }
        let indices: Vec<i32> = bind.results.iter().map(|r| r.0).collect();
        let scores: Vec<f32> = bind.results.iter().map(|r| r.1).collect();
        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(indices)),
                Arc::new(Float32Array::from(scores)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ])
    }
}
