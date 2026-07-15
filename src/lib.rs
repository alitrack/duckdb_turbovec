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

// ── turboquant_build(vectors_str, dim, bit_width, output_path) ──

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
                "turboquant_build: expected 4 params (vectors_str, dim, bit_width, output_path)"
                    .into(),
            );
        }
        let vectors_str = bind.get_parameter(0).to_string();
        let dim: usize = bind.get_parameter(1).to_string().parse()?;
        let bw: usize = bind.get_parameter(2).to_string().parse()?;
        let output_path = bind.get_parameter(3).to_string();

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

// ── Entrypoint ──

#[duckdb_entrypoint_c_api(ext_name = "turbovec")]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<TurboQuantSearchVTab>("turboquant_search")?;
    con.register_table_function::<TurboQuantScoreVTab>("turboquant_score")?;
    con.register_table_function::<TurboQuantBuildVTab>("turboquant_build")?;
    Ok(())
}
