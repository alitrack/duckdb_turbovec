use arrow::{
    array::{Float32Array, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{arrow::record_batch_to_duckdb_data_chunk, BindInfo, InitInfo, TableFunctionInfo, VTab},
    Connection, Result,
};
use std::{
    error::Error,
    sync::{atomic::AtomicPtr, Arc},
};
use turbovec::TurboQuantIndex;

// ── Global: raw DuckDB database handle (captured at LOAD time) ──

static RAW_DB: AtomicPtr<std::ffi::c_void> = AtomicPtr::new(std::ptr::null_mut());

fn temp_conn() -> duckdb::Result<Connection> {
    let raw = RAW_DB.load(std::sync::atomic::Ordering::Relaxed);
    if raw.is_null() {
        panic!("turboquant_build: extension not loaded properly");
    }
    unsafe { Connection::open_from_raw(raw as duckdb::ffi::duckdb_database) }
}

// ── turboquant_search(index_path, query_str, k) ──

#[repr(C)]
struct SearchInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
struct SearchBindData {
    results: Vec<(i32, f32)>,
    schema: Arc<arrow::datatypes::Schema>,
}

struct TurboQuantSearchVTab;

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

// ── turboquant_build(table, col, bit_width, output_path) ──

struct TurboQuantBuildVTab;

#[repr(C)]
struct BuildInitData {
    done: std::sync::atomic::AtomicBool,
}

#[repr(C)]
struct BuildBindData {
    output_path: String,
    rows: usize,
}

impl VTab for TurboQuantBuildVTab {
    type BindData = BuildBindData;
    type InitData = BuildInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 4 {
            return Err("turboquant_build: expected 4 params (table, col, bit_width, output_path)".into());
        }
        let table = bind.get_parameter(0).to_string();
        let col = bind.get_parameter(1).to_string();
        let bw: usize = bind.get_parameter(2).to_string().parse()?;
        let output_path = bind.get_parameter(3).to_string();

        let con = temp_conn()?;
        let dim_sql = format!("SELECT len({col}) FROM {table} LIMIT 1");
        let dim: usize = con
            .query_row(&dim_sql, [], |row| row.get::<_, i32>(0))
            .map_err(|e| format!("turboquant_build: cannot read '{table}': {e}"))?
            as usize;
        let count_sql = format!("SELECT count(*) FROM {table}");
        let n: usize = con
            .query_row(&count_sql, [], |row| row.get::<_, i64>(0))
            .map_err(|e| format!("turboquant_build: {e}"))? as usize;
        if n == 0 {
            return Err("turboquant_build: table has no rows".into());
        }

        let read_sql = format!("SELECT {col} FROM {table}");
        let mut stmt = con.prepare(&read_sql)?;
        let mut rows = stmt.query([])?;

        let mut idx =
            TurboQuantIndex::new(dim, bw).map_err(|e| format!("turboquant_build: {e}"))?;
        let mut batch = Vec::with_capacity(2048 * dim);
        while let Some(row) = rows.next()? {
            let vec_str: String = row.get(0)?;
            let vec = parse_float_array(&vec_str)?;
            batch.extend_from_slice(&vec);
            if batch.len() >= 2048 * dim {
                idx.add(&batch);
                batch.clear();
            }
        }
        if !batch.is_empty() {
            idx.add(&batch);
        }
        idx.write(&output_path)
            .map_err(|e| format!("turboquant_build: write error: {e}"))?;

        bind.add_result_column("output_path", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("rows", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(BuildBindData { output_path, rows: n })
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
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
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

// ── Entrypoint ──

unsafe fn extension_init(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<TurboQuantSearchVTab>("turboquant_search")?;
    con.register_table_function::<TurboQuantBuildVTab>("turboquant_build")?;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn turbovec_init_c_api(raw_db: duckdb::ffi::duckdb_database) {
    RAW_DB.store(
        raw_db as *mut std::ffi::c_void,
        std::sync::atomic::Ordering::Relaxed,
    );
    let con = Connection::open_from_raw(raw_db).unwrap();
    extension_init(con).unwrap();
}
