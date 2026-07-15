use arrow::{
    array::{Float32Array, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    duckdb_entrypoint_c_api,
    vtab::{
        arrow::record_batch_to_duckdb_data_chunk,
        BindInfo, InitInfo, TableFunctionInfo, VTab,
    },
    Connection, Result,
};
use std::{error::Error, sync::Arc};
use turbovec::TurboQuantIndex;

/// Per-scan state: loaded index + search results.
#[repr(C)]
struct SearchInitData {
    done: std::sync::atomic::AtomicBool,
}

/// Bound parameters stored for func() access.
#[repr(C)]
struct SearchBindData {
    index_path: String,
    query: Vec<f32>,
    k: usize,
    /// Search results: rows of (idx: i32, score: f32).
    results: Vec<(i32, f32)>,
    /// Output schema for Arrow batch construction.
    schema: Arc<arrow::datatypes::Schema>,
}

struct TurboQuantSearchVTab;

impl VTab for TurboQuantSearchVTab {
    type BindData = SearchBindData;
    type InitData = SearchInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let param_count = bind.get_parameter_count();
        if param_count < 3 {
            return Err(format!(
                "turboquant_search: expected 3 params (path, query::FLOAT[N], k), got {param_count}"
            )
            .into());
        }

        let index_path = bind.get_parameter(0).to_string();

        let query_str = bind.get_parameter(1).to_string();
        let query = parse_float_array(&query_str)?;

        let k: usize = bind
            .get_parameter(2)
            .to_string()
            .parse()
            .map_err(|e| format!("turboquant_search: invalid k: {e}"))?;

        // Load index and run search eagerly in bind
        let index = TurboQuantIndex::load(&index_path)
            .map_err(|e| format!("turboquant_search: cannot load '{index_path}': {e}"))?;

        let sr = index.search(&query, k);

        // Flatten results: one query, k results
        let mut results = Vec::with_capacity(sr.nq * k);
        for qi in 0..sr.nq {
            let offset = qi * k;
            for j in 0..k {
                let idx = sr.indices[offset + j] as i32;
                let score = sr.scores[offset + j];
                results.push((idx, score));
            }
        }

        // Declare output schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));

        // Register output columns with DuckDB
        bind.add_result_column(
            "idx",
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        );
        bind.add_result_column(
            "score",
            LogicalTypeHandle::from(LogicalTypeId::Float),
        );

        Ok(SearchBindData {
            index_path,
            query,
            k,
            results,
            schema,
        })
    }

    fn init(_info: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(SearchInitData {
            done: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        let bind = func.get_bind_data();

        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }

        let n = bind.results.len().min(2048);

        if n == 0 {
            init.done.store(true, std::sync::atomic::Ordering::Relaxed);
            output.set_len(0);
            return Ok(());
        }

        // Build Arrow RecordBatch from results
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
            LogicalTypeHandle::from(LogicalTypeId::Varchar), // index_path
            LogicalTypeHandle::from(LogicalTypeId::Varchar), // query (FLOAT[N] as string)
            LogicalTypeHandle::from(LogicalTypeId::Integer), // k
        ])
    }
}

/// Parse a DuckDB float array string like "[1.0, 2.0, 3.0]".
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

#[duckdb_entrypoint_c_api(ext_name = "turbovec")]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<TurboQuantSearchVTab>("turboquant_search")?;
    Ok(())
}
