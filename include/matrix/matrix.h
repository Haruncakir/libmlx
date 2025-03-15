#ifndef MATRIX_H
#define MATRIX_H

#include "matpool.h"

typedef struct {
    size_t row;
    size_t col;
    size_t stride; /* Stride between rows (for padding) */
    float *data;   /* keep in mind: NVIDIA GPUs with tensor cores and
                      many deep learning models can tolerate reduced precision (float16) */
} mat_t;

#ifdef __SIZE_TYPE__
typedef __SIZE_TYPE__ size_t;
#else
typedef unsigned long size_t;
#endif

/* parameter annotations for better static analysis */
#define MAT_IN
#define MAT_OUT
#define MAT_INOUT
#define MAT_NULLABLE
#define MAT_NONNULL

typedef unsigned char mat_status_t;

/* Matrix status list */
#define MATRIX_SUCCESS               ((mat_status_t)0)
#define MATRIX_NOT_INITIALIZED       ((mat_status_t)1)
#define MATRIX_NULL_POINTER          ((mat_status_t)2)
#define MATRIX_REGION_FULL           ((mat_status_t)3)
#define MATRIX_INVALID_REGION        ((mat_status_t)4)
#define MATRIX_DIMENSION_MISMATCH    ((mat_status_t)5)
#define MATRIX_UNSUPPORTED_OPERATION ((mat_status_t)6)

/*
4 main operations
+----------------------------------------------------------------------------------------------------+
|  Operation                             |  Description	                 |  Used in ML?              |
+----------------------------------------+-------------------------------+---------------------------+
|1-  Matrix Multiplication (A x B)       |  Combining two matrices	     |  Yes, core of neural nets |
|2-  Dot Product (A . B)                 |  Sum of element-wise products |  Yes, in dense layers     |
|3-  Element-wise Addition (A + B)       |  Adding two matrices          |  Yes, in bias addition    |
|4-  Element-wise Multiplication (A ⊙ B) |  Multiplying element-wise     |  Yes, in activations      |
+----------------------------------------------------------------------------------------------------+
*/

// Design choice -> caller-provided buffers.

/* Core matrix operations - implementation varies by architecture */
mat_status_t matadd       (const mat_t MAT_IN *a, const mat_t MAT_IN *b, mat_t MAT_OUT *c);
mat_status_t matsub       (const mat_t MAT_IN *a, const mat_t MAT_IN *b, mat_t MAT_OUT *c);
mat_status_t matmul       (const mat_t MAT_IN *a, const mat_t MAT_IN *b, mat_t MAT_OUT *c);
mat_status_t matvecmul    (const mat_t MAT_IN *a, const float MAT_IN *x, float MAT_OUT *y);
mat_status_t mattranspose (const mat_t MAT_IN *a,                        mat_t MAT_OUT *b);
mat_status_t matreshape   (const mat_t MAT_IN *a, size_t row, size_t col, mat_t MAT_OUT *b);
mat_status_t matreshapeip (const mat_t MAT_INOUT *a, size_t row, size_t col);
mat_status_t matreshapeblock(const mat_t MAT_IN *a, size_t row, size_t col, mat_t MAT_OUT *b);
mat_status_t matreshapereg(mat_region_t *reg, const mat_t MAT_IN *a, 
                               size_t row, size_t col, mat_t MAT_OUT *b);
float        matdot       (const float *a,        const float *b,        size_t length);

/* Common utility functions - same implementation for all architectures */
float        matget       (const mat_t *m, size_t row, size_t col);
void         matset       (mat_t *m, size_t row, size_t col, float value);
mat_status_t matfill      (mat_t *m, float value);
float*       matrowptr    (mat_t *m, size_t row);
mat_status_t matcopy      (const mat_t *src, mat_t *dst);
mat_status_t matidentity  (mat_t *m);
float        matnorm      (const mat_t *m);
// void matprint(const mat_t *m);

#endif // MATRIX_H