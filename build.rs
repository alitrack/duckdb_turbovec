fn main() {
    // Build a stub static library named 'openblas' that provides CBLAS symbols.
    // cblas-sys's build.rs adds `-lopenblas` — we satisfy it with our own stub.
    cc::Build::new()
        .file("src/cblas_stub.c")
        .compile("openblas");
}
