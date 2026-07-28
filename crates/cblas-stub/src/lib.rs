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
    if m <= 0 || n <= 0 || k <= 0 { return; }
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);
    let max_a = if k > lda { k * lda } else { m * k };
    let max_b = if n > ldb { n * ldb } else { k * n };
    let max_c = m * n;
    unsafe {
        let a_slice = std::slice::from_raw_parts(a, max_a);
        let b_slice = std::slice::from_raw_parts(b, max_b);
        let c_slice = std::slice::from_raw_parts_mut(c, max_c);
        dgemm(m, k, n, alpha, a_slice, lda, 1, b_slice, ldb, 1, beta, c_slice, ldc, 1);
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
