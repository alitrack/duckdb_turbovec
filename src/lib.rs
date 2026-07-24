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
use turbovec::TurboQuantIndex;

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
        let idx = TurboQuantIndex::load(&path)
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
        let n = bind.results.len().min(2048);
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

        let idx = TurboQuantIndex::load(&path)
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
        let n = bind.results.len().min(2048);
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

/// K-means clustering (Lloyd's algorithm)
fn kmeans(
    vectors: &[f32], dim: usize, k: usize, max_iters: usize,
) -> (Vec<f32>, Vec<usize>) {
    let n = vectors.len() / dim;
    let mut centroids = vec![0.0f32; k * dim];
    let mut assignments = vec![0usize; n];

    // Init: pick first k vectors as centroids
    let step = (n / k).max(1);
    for ci in 0..k {
        let vi = (ci * step).min(n - 1);
        centroids[ci * dim..(ci + 1) * dim]
            .copy_from_slice(&vectors[vi * dim..(vi + 1) * dim]);
    }

    for _iter in 0..max_iters {
        // Assign each vector to nearest centroid
        let mut changed = false;
        for vi in 0..n {
            let v = &vectors[vi * dim..(vi + 1) * dim];
            let mut best_c = assignments[vi];
            let mut best_d = l2_dist(v, &centroids[best_c * dim..(best_c + 1) * dim]);
            for ci in 0..k {
                let d = l2_dist(v, &centroids[ci * dim..(ci + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best_c = ci;
                }
            }
            if best_c != assignments[vi] {
                changed = true;
                assignments[vi] = best_c;
            }
        }
        if !changed { break; }

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
        let num_lists: usize = bind.get_parameter(3).to_string().parse()?;
        let output_dir = bind.get_parameter(4).to_string();

        let vectors = parse_nested_float_arrays(&vectors_str, dim)?;
        let n = vectors.len() / dim;
        if n == 0 {
            return Err("turboquant_build_ivf: no vectors provided".into());
        }
        let lists = num_lists.min(n);
        if lists < 1 {
            return Err("turboquant_build_ivf: num_lists must be >= 1".into());
        }

        // K-means clustering
        let (centroids, assignments) = kmeans(&vectors, dim, lists, 20);

        // Build per-cluster indices and id-maps (global → cluster-local)
        let mut cluster_vecs: Vec<Vec<f32>> = vec![Vec::new(); lists];
        let mut id_maps: Vec<Vec<i32>> = vec![Vec::new(); lists];
        for vi in 0..n {
            let c = assignments[vi];
            id_maps[c].push(vi as i32);
            cluster_vecs[c].extend_from_slice(&vectors[vi * dim..(vi + 1) * dim]);
        }

        std::fs::create_dir_all(&output_dir)?;
        for ci in 0..lists {
            if cluster_vecs[ci].is_empty() { continue; }
            let mut idx = TurboQuantIndex::new(dim, bw)
                .map_err(|e| format!("turboquant_build_ivf cluster {ci}: {e}"))?;
            idx.add(&cluster_vecs[ci]);
            idx.write(&format!("{}/cluster_{ci}.tv", output_dir))
                .map_err(|e| format!("turboquant_build_ivf write {ci}: {e}"))?;

            // Write idmap: local_pos → global_idx
            let idmap_path = format!("{}/idmap_{ci}.bin", output_dir);
            let bytes: Vec<u8> = id_maps[ci].iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(&idmap_path, bytes)?;
        }

        // Write meta
        let meta = IvfMeta { dim, bit_width: bw, num_lists: lists, centroids };
        meta.write(&format!("{}/meta.bin", output_dir))?;

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

        let meta = IvfMeta::load(&format!("{}/meta.bin", dir))?;
        let dim = meta.dim;
        // probes=0 or probes=-1 → auto: scan all clusters (guaranteed recall)
        let probes = if probes == 0 || probes == usize::MAX {
            meta.num_lists
        } else {
            probes.min(meta.num_lists)
        };

        // Find nearest centroids
        let mut centroid_dists: Vec<(usize, f32)> = (0..meta.num_lists)
            .map(|ci| {
                let c = &meta.centroids[ci * dim..(ci + 1) * dim];
                (ci, l2_dist(&query, c))
            })
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Search top probes clusters
        let mut all: Vec<(i32, f32)> = Vec::new();
        for (ci, _) in centroid_dists.iter().take(probes) {
            let path = format!("{}/cluster_{ci}.tv", dir);
            if let Ok(idx) = TurboQuantIndex::load(&path) {
                let n = idx.len();
                if n == 0 { continue; }

                // Load idmap: local_pos → global_idx
                let idmap = load_idmap(&format!("{}/idmap_{ci}.bin", dir));

                let sr = idx.search(&query, k.min(n));
                for j in 0..k.min(n) {
                    let local = sr.indices[j] as usize;
                    let global = if local < idmap.len() { idmap[local] } else { local as i32 };
                    all.push((global, sr.scores[j]));
                }
            }
        }

        // Sort and take top-k
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        let n = bind.results.len().min(2048);
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

// ── Entrypoint ──

#[duckdb_entrypoint_c_api(ext_name = "turbovec")]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<TurboQuantSearchVTab>("turboquant_search")?;
    con.register_table_function::<TurboQuantScoreVTab>("turboquant_score")?;
    con.register_table_function::<TurboQuantBuildVTab>("turboquant_build")?;
    con.register_table_function::<TurboQuantBuildListVTab>("turboquant_build_list")?;
    con.register_table_function::<TurboQuantBuildConcatVTab>("turboquant_build_concat")?;
    con.register_table_function::<TurboQuantAddVTab>("turboquant_add")?;
    con.register_table_function::<TurboQuantRemoveVTab>("turboquant_remove")?;
    con.register_table_function::<IvfBuildVTab>("turboquant_build_ivf")?;
    con.register_table_function::<IvfSearchVTab>("turboquant_search_ivf")?;
    Ok(())
}
