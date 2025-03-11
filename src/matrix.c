#include "matrix.h"

// TODO: optimize for SIMD instructions.

mat_status_t matmul(const mat_t __INPUT *A, const mat_t __INPUT *B, mat_t __OUTPUT *C) {
    if (!A || !B || !C)
        return MATRIX_NULL_POINTER;
    
    if (A->col != B->row || C->row != A->row || C->col != B->col)
        return MATRIX_DIMENSION_MISMATCH;
    
    size_t i = 0, j = 0, k = 0;
    for (; i < A->row; ++i) {
        for(; j < B->col; ++j) {
            C->data[i * C->col + j] = 0.0f;
            for (; k < A->col; ++k)
                C->data[i * C->col + j] += A->data[i * A->col + k] * B->data[k * B->col + j];
        }
    }

    return MATRIX_SUCCESS;
}

/**
 * Element-wise multiplication: C = A ⊙ B
 * A, B, and C must have the same dimensions
 */
mat_status_t matemul(const mat_t *A, const mat_t *B, mat_t *C) {
    if (!A || !B || !C)
        return MATRIX_NULL_POINTER;
    
    // Check dimensions match
    if (A->row != B->row || A->col != B->col || 
        C->row != A->row || C->col != A->col)
        return MATRIX_DIMENSION_MISMATCH;
    
    size_t total_elements = A->row * A->col, i = 0;
    
    for (; i < total_elements; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
    
    return MATRIX_SUCCESS;
}

/**
 * Dot product: C = A · B
 * A and B must be vectors of the same length
 * C is a 1x1 matrix (scalar)
 */
mat_status_t matdot(const mat_t *A, const mat_t *B, mat_t *C) {
    if (!A || !B || !C)
        return MATRIX_NULL_POINTER;
    
    // Check that A and B are vectors with the same length
    size_t len_a = A->row * A->col;
    size_t len_b = B->row * B->col;
    
    if (len_a != len_b)
        return MATRIX_DIMENSION_MISMATCH;
    
    // Check that C is a 1x1 matrix
    if (C->row != 1 || C->col != 1)
        return MATRIX_DIMENSION_MISMATCH;
    
    float sum = 0.0f;
    size_t i = 0;
    for (; i < len_a; i++) {
        sum += A->data[i] * B->data[i];
    }
    
    C->data[0] = sum;
    
    return MATRIX_SUCCESS;
}

/**
 * Matrix addition: C = A + B
 * A, B, and C must have the same dimensions
 */
mat_status_t matadd(const mat_t *A, const mat_t *B, mat_t *C) {
    if (!A || !B || !C)
        return MATRIX_NULL_POINTER;
    
    // Check dimensions match
    if (A->row != B->row || A->col != B->col || 
        C->row != A->row || C->col != A->col)
        return MATRIX_DIMENSION_MISMATCH;
    
    size_t total_elements = A->row * A->col, i = 0;
    
    for (; i < total_elements; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
    
    return MATRIX_SUCCESS;
}