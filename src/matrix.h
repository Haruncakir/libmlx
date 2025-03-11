/**
 * Author - Harun Cakir
 */

/**
 * HEADS UP
 * - Use caller provided buffers e.g.
 *     mlx_result_t mlx_infer(mlx_model_t *model, float *input, size_t input_size, float *output);
 * - Avoid Temporary Buffers in Computation
 * - Use SIMD Registers Instead of Extra Memory
 */

#ifndef MATRIX_H
#define MATRIX_H

/*
There should be NO unnecessary library dependency.
*/
#include "def.h"

/*
Assume that I have 4x3 matrix
M = [
    [1 2 3]
    [1 2 3]
    [1 2 3]
    [1 2 3]
]
I can flatten this matrix into 1D array.
How can I know where a row ends ?
Since I know the column number in this case it is 3.
M = [1 2 3 | 1 2 3 | 1 2 3 | 1 2 3]
*/

/*
What are the main operations?
+----------------------------------------------------------------------------------------------------+
|  Operation                             |  Description	                 |  Used in ML?              |
+----------------------------------------+-------------------------------+---------------------------+
|1-  Matrix Multiplication (A x B)       |  Combining two matrices	     |  Yes, core of neural nets |
|2-  Dot Product (A . B)                 |  Sum of element-wise products |  Yes, in dense layers     |
|3-  Element-wise Addition (A + B)       |  Adding two matrices          |  Yes, in bias addition    |
|4-  Element-wise Multiplication (A ⊙ B) |  Multiplying element-wise     |  Yes, in activations      |
+----------------------------------------------------------------------------------------------------+
*/

// Naming convention is like standard c library functions.
// Design choice -> caller-provided buffers.
// Consider: #define f(a, b, output) assert(a->col == b->row); output = func_(a, b)
mat_status_t matmul(const mat_t __INPUT *A, const mat_t __INPUT *B, mat_t __OUTPUT *C);
mat_status_t matemul(const mat_t __INPUT *A, const mat_t __INPUT *B, mat_t __OUTPUT *C);
mat_status_t matdot(const mat_t __INPUT *A, const mat_t __INPUT *B, mat_t __OUTPUT *C);
mat_status_t matadd(const mat_t __INPUT *A, const mat_t __INPUT *B, mat_t __OUTPUT *C);

#endif // MATRIX_H