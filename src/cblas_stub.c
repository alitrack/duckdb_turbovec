// cblas_stub.c — provides all CBLAS symbols as no-ops.
// Eliminates the need for system OpenBLAS at link time.
#include <stdint.h>

// Complex types
typedef double c_double_complex[2];
typedef float  c_float_complex[2];

// Enums
typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
typedef enum { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
typedef enum { CblasLeft=141, CblasRight=142 } CBLAS_SIDE;

// Level 1 stubs
void cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy, float *r) {}
void cblas_dsdot(const int n, const float *x, const int incx, const float *y, const int incy, double *r) {}
float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy) { return 0; }
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) { return 0; }
float cblas_snrm2(const int n, const float *x, const int incx) { return 0; }
float cblas_sasum(const int n, const float *x, const int incx) { return 0; }
double cblas_dnrm2(const int n, const double *x, const int incx) { return 0; }
double cblas_dasum(const int n, const double *x, const int incx) { return 0; }
int cblas_isamax(const int n, const float *x, const int incx) { return 0; }
int cblas_idamax(const int n, const double *x, const int incx) { return 0; }
void cblas_sswap(const int n, float *x, const int incx, float *y, const int incy) {}
void cblas_scopy(const int n, const float *x, const int incx, float *y, const int incy) {}
void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy) {}
void cblas_dswap(const int n, double *x, const int incx, double *y, const int incy) {}
void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy) {}
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) {}
void cblas_cswap(const int n, void *x, const int incx, void *y, const int incy) {}
void cblas_ccopy(const int n, const void *x, const int incx, void *y, const int incy) {}
void cblas_caxpy(const int n, const void *alpha, const void *x, const int incx, void *y, const int incy) {}
void cblas_zswap(const int n, void *x, const int incx, void *y, const int incy) {}
void cblas_zcopy(const int n, const void *x, const int incx, void *y, const int incy) {}
void cblas_zaxpy(const int n, const void *alpha, const void *x, const int incx, void *y, const int incy) {}
void cblas_scnrm2(const int n, const void *x, const int incx, float *r) {}
void cblas_scasum(const int n, const void *x, const int incx, float *r) {}
void cblas_dznrm2(const int n, const void *x, const int incx, double *r) {}
void cblas_dzasum(const int n, const void *x, const int incx, double *r) {}
int cblas_icamax(const int n, const void *x, const int incx) { return 0; }
int cblas_izamax(const int n, const void *x, const int incx) { return 0; }

// Level 2 stubs
void cblas_sgemv(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE t, const int m, const int n,
    const float a, const float *A, const int lda, const float *x, const int incx,
    const float b, float *y, const int incy) {}
void cblas_dgemv(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE t, const int m, const int n,
    const double a, const double *A, const int lda, const double *x, const int incx,
    const double b, double *y, const int incy) {}
void cblas_cgemv(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE t, const int m, const int n,
    const void *a, const void *A, const int lda, const void *x, const int incx,
    const void *b, void *y, const int incy) {}
void cblas_zgemv(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE t, const int m, const int n,
    const void *a, const void *A, const int lda, const void *x, const int incx,
    const void *b, void *y, const int incy) {}

// Level 3 stubs
void cblas_sgemm(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE ta, const CBLAS_TRANSPOSE tb,
    const int m, const int n, const int k, const float a, const float *A, const int lda,
    const float *B, const int ldb, const float b, float *C, const int ldc) {}
void cblas_dgemm(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE ta, const CBLAS_TRANSPOSE tb,
    const int m, const int n, const int k, const double a, const double *A, const int lda,
    const double *B, const int ldb, const double b, double *C, const int ldc) {}
void cblas_cgemm(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE ta, const CBLAS_TRANSPOSE tb,
    const int m, const int n, const int k, const void *a, const void *A, const int lda,
    const void *B, const int ldb, const void *b, void *C, const int ldc) {}
void cblas_zgemm(const CBLAS_LAYOUT l, const CBLAS_TRANSPOSE ta, const CBLAS_TRANSPOSE tb,
    const int m, const int n, const int k, const void *a, const void *A, const int lda,
    const void *B, const int ldb, const void *b, void *C, const int ldc) {}

// Misc
double cblas_dcabs1(const double *z) { return 0; }
float  cblas_scabs1(const float  *c) { return 0; }
void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *r) {}
void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *r) {}
void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *r) {}
void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *r) {}
