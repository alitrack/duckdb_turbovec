LOAD '/mnt/d/wsl2/duckdb_turbovec/build/release/extension/turbovec/turbovec.duckdb_extension';
SET lambda_syntax='ENABLE_SINGLE_ARROW';

CREATE OR REPLACE TABLE items AS
SELECT i::BIGINT AS id, (i % 1000)::INTEGER AS cat,
       list_transform(range(64), j -> round(10*random(), 3))::FLOAT[64] AS emb
FROM range(20000) t(i);

SET VARIABLE vv = (SELECT '[' || string_agg('[' || array_to_string(list_transform(emb, x -> x::VARCHAR), ',') || ']', '') FROM items);
SELECT * FROM turboquant_build_ivf(getvariable('vv'), 64, 4, 0, '/tmp/tvfilter_test/', true);

SET VARIABLE qs = (SELECT array_to_string(list_transform(list_transform(range(64), j -> 0.5), x -> x::VARCHAR), ','));
SET VARIABLE qarr = list_transform(range(64), j -> 0.5::FLOAT)::FLOAT[64];
SET VARIABLE filt = (SELECT list(id) FROM items WHERE cat = 42);

SELECT '-- FILTERED plain --' AS marker;
SELECT * FROM turboquant_search_ivf_filter('/tmp/tvfilter_test/', getvariable('qs'), 10, getvariable('filt'), 0, 0);
SELECT '-- EXACT GT --' AS marker;
CREATE OR REPLACE TABLE gt AS
SELECT row_number() OVER () AS rk, id FROM items WHERE cat = 42 ORDER BY array_distance(emb, getvariable('qarr')) LIMIT 10;
SELECT * FROM gt ORDER BY rk;
SELECT '-- rerank x8 overlap --' AS marker;
SELECT count(*) AS overlap FROM (
  SELECT idx AS id FROM turboquant_search_ivf_filter('/tmp/tvfilter_test/', getvariable('qs'), 10, getvariable('filt'), 0, 8)
) a SEMI JOIN gt ON a.id = gt.id;
SELECT '-- overlap plain --' AS marker;
SELECT count(*) AS overlap FROM (
  SELECT idx AS id FROM turboquant_search_ivf_filter('/tmp/tvfilter_test/', getvariable('qs'), 10, getvariable('filt'), 0, 0)
) a SEMI JOIN gt ON a.id = gt.id;
