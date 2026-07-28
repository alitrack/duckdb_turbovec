//! Drop-in pure-Rust replacement for cblas-sys.
//! Provides cblas_dgemm using matrixmultiply.

use matrixmultiply::dgemm;

#[no_mangle]
pub unsafe extern "C" fn cblas_dgemm(
    _order: i32, _transa: i32, _transb: i32,
    m: i32, n: i32, k: i32,
    alpha: f64, a: *const f64, lda: i32,
    b: *const f64, ldb: i32,
    beta: f64, c: *mut f64, ldc: i32,
) {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    if m == 0 || n == 0 || k == 0 { return; }
    let l1 = lda as usize;
    let l2 = ldb as usize;
    let l3 = ldc as usize;
    unsafe {
        dgemm(m, k, n, alpha,
            std::slice::from_raw_parts(a, m * l1.max(k)), l1, 1,
            std::slice::from_raw_parts(b, k * l2.max(n)), l2, 1,
            beta,
            std::slice::from_raw_parts_mut(c, m * l3.max(n)), l3, 1);
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_sgemm(
    _order: i32, _transa: i32, _transb: i32,
    _m: i32, _n: i32, _k: i32,
    _alpha: f32, _a: *const f32, _lda: i32,
    _b: *const f32, _ldb: i32,
    _beta: f32, _c: *mut f32, _ldc: i32,
) {}
