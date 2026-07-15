use duckdb::{params, Connection};
use turbovec::{TurboQuantBuildVTab, TurboQuantScoreVTab, TurboQuantSearchVTab};

#[test]
fn test_build_search_score_roundtrip() {
    let conn = Connection::open_in_memory().unwrap();

    // NOTE: extension_entrypoint is the C API entrypoint for loadable extensions.
    // In tests we register functions directly instead.
    conn.register_table_function::<TurboQuantSearchVTab>("turboquant_search")
        .unwrap();
    conn.register_table_function::<TurboQuantScoreVTab>("turboquant_score")
        .unwrap();
    conn.register_table_function::<TurboQuantBuildVTab>("turboquant_build")
        .unwrap();

    conn.execute_batch(
        "CREATE TABLE docs AS
         SELECT * FROM (VALUES
             (1, '[1.0, 0.0, 0.0, 0.0]'::VARCHAR),
             (2, '[0.0, 1.0, 0.0, 0.0]'::VARCHAR),
             (3, '[0.0, 0.0, 1.0, 0.0]'::VARCHAR)
         ) t(id, emb);",
    )
    .unwrap();

    let vectors_str: String = conn
        .query_row("SELECT string_agg(emb, ',') FROM docs", params![], |row| {
            row.get(0)
        })
        .unwrap();
    let path = "/tmp/test_duckdb_turbovec_roundtrip.tv";

    // 1. Build
    let sql = format!(
        "SELECT * FROM turboquant_build('{}', 4, 4, '{}')",
        vectors_str, path
    );
    let mut stmt = conn.prepare(&sql).unwrap();
    let mut rows = stmt.query(params![]).unwrap();
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<_, String>(0).unwrap(), path);
    assert_eq!(row.get::<_, i32>(1).unwrap(), 3);
    drop(rows);

    // 2. Score
    let sql = format!(
        "SELECT * FROM turboquant_score('{}', '[1.0, 0.0, 0.0, 0.0]')",
        path
    );
    let mut stmt = conn.prepare(&sql).unwrap();
    let mut rows = stmt.query(params![]).unwrap();
    let mut scores = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        scores.push((row.get::<_, i32>(0).unwrap(), row.get::<_, f32>(1).unwrap()));
    }
    assert_eq!(scores.len(), 3);
    for w in scores.windows(2) {
        assert!(w[0].1 >= w[1].1, "not sorted: {:?}", scores);
    }
    assert_eq!(scores[0].0, 0, "doc 0 should be top");

    // 3. Search consistency
    let sql = format!(
        "SELECT * FROM turboquant_search('{}', '[1.0, 0.0, 0.0, 0.0]', 2)",
        path
    );
    let mut stmt = conn.prepare(&sql).unwrap();
    let mut rows = stmt.query(params![]).unwrap();
    let r1 = rows.next().unwrap().unwrap();
    assert_eq!(r1.get::<_, i32>(0).unwrap(), 0);
    assert!((r1.get::<_, f32>(1).unwrap() - scores[0].1).abs() < 1e-5);

    std::fs::remove_file(path).ok();
}
