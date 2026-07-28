fn main() {
    // Build a stub static library that provides CBLAS symbols.
    // This prevents the need for system libopenblas.
    cc::Build::new()
        .file("src/cblas_stub.c")
        .compile("cblas_stub");
    println!("cargo:rustc-link-lib=static=cblas_stub");
}
