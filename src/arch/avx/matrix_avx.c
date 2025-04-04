#include "../../../include/matrix/matrix_config.h"
#include "../../../include/matrix/matrix.h"

// #if defined(MAT_USE_AVX)
#include <immintrin.h>

mat_status_t matadd(const mat_t *a, const mat_t *b, mat_t *c) {
    if (!a || !b || !c) {
        return MATRIX_NULL_POINTER;
    }
    
    if (a->row != b->row || a->col != b->col ||
        c->row != a->row || c->col != a->col) {
        return MATRIX_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < a->row; ++i) {
        float *a_row = &a->data[i * a->stride];
        float *b_row = &b->data[i * b->stride];
        float *c_row = &c->data[i * c->stride];

        size_t j = 0;
        for (; j + 7 < a->col; j+=8) {
            __m256 va = _mm256_loadu_ps(&a_row[j]);
            __m256 vb = _mm256_loadu_ps(&b_row[j]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&c_row[j], vc);
        }

        // handle remaining elements
        for(; j < a->col; ++j) {
            c_row[j] = a_row[j] + b_row[j];
        }
    }

    return MATRIX_SUCCESS;
}

mat_status_t matsub(const mat_t *a, const mat_t *b, mat_t *c) {
    if (!a || !b || !c) {
        return MATRIX_NULL_POINTER;
    }
    
    if (a->row != b->row || a->col != b->col ||
        c->row != a->row || c->col != a->col) {
        return MATRIX_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < a->row; ++i) {
        float *a_row = &a->data[i * a->stride];
        float *b_row = &b->data[i * b->stride];
        float *c_row = &c->data[i * c->stride];

        size_t j = 0;
        for (; j + 7 < a->col; j+=8) {
            __m256 va = _mm256_loadu_ps(&a_row[j]);
            __m256 vb = _mm256_loadu_ps(&b_row[j]);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(&c_row[j], vc);
        }

        // handle remaining elements
        for(; j < a->col; ++j) {
            c_row[j] = a_row[j] - b_row[j];
        }
    }

    return MATRIX_SUCCESS;
}

mat_status_t matmul(const mat_t *a, const mat_t *b, mat_t *c) {
    if (!a || !b || !c) {
        return MATRIX_NULL_POINTER;
    }
    
    if (a->col != b->row || c->row != a->row || c->col != b->col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    /* Initialize C to zeros */
    for (size_t i = 0; i < c->row; ++i) {
        for (size_t j = 0; j < c->col; ++j) {
            c->data[i * c->stride + j] = 0.0f;
        }
    }
    
    for (size_t i = 0; i < a->row; ++i) {
        for (size_t k = 0; k < a->col; ++k) {
            __m256 va = _mm256_set1_ps(matget(a, i, k));
            size_t j = 0;

            for(; j + 7 < b->col; j+=8) {
                float *c_row = matrowptr(c, i);
                float *b_row = &b->data[k * b->stride];

                __m256 vb = _mm256_loadu_ps(&b_row[j]);
                __m256 vc = _mm256_loadu_ps(&c_row[j]);
                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                _mm256_storeu_ps(&c_row[j], vc);
            }

            // handle remaining cols
            for (; j < b->col; ++j) {
                float c_val = matget(c, i, j);
                float a_val = matget(a, i, k);
                float b_val = matget(b, k, j);
                matset(c, i, j, c_val + a_val * b_val);
            }
        }
    }

    return MATRIX_SUCCESS;
}

mat_status_t mattranspose (const mat_t *a, mat_t *b) {
    if (!a || !b) {
        return MATRIX_NULL_POINTER;
    }
    
    if (b->row != a->col || b->col != a->row) {
        return MATRIX_DIMENSION_MISMATCH;
    }

    // scalar implementation for small matrices
    if (a->row < 8 || a->col < 8) {
        for (size_t i = 0; i < a->row; i++) {
            for (size_t j = 0; j < a->col; j++) {
                float val = a->data[i * a->stride + j];
                b->data[j * b->stride + i] = val;
            }
        }
        return MATRIX_SUCCESS;
    }

    // For larger matrices, use AVX to transpose blocks
    // Process the matrix in 8x8 blocks where possible
    const size_t block_size = 8; // 8 floats fit in an AVX register
    
    // First, handle complete 8x8 blocks
    for (size_t i = 0; i + block_size <= a->row; i += block_size) {
        for (size_t j = 0; j + block_size <= a->col; j += block_size) {
            // Load 8 rows of 8 elements each
            __m256 rows[8];
            for (size_t k = 0; k < block_size; k++) {
                rows[k] = _mm256_loadu_ps(&a->data[(i + k) * a->stride + j]);
            }
            
            // Transpose 8x8 block using AVX intrinsics
            __m256 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
            __m256 out0, out1, out2, out3, out4, out5, out6, out7;
            
            // Interleave 32-bit elements
            // r0 = [a0, a1, a2, a3, a4, a5, a6, a7]
            // r1 = [b0, b1, b2, b3, b4, b5, b6, b7]
            // ...
            // r7 = [h0, h1, h2, h3, h4, h5, h6, h7]
            
            temp0 = _mm256_unpacklo_ps(rows[0], rows[1]); // [a0,b0,a2,b2,a4,b4,a6,b6]
            temp1 = _mm256_unpackhi_ps(rows[0], rows[1]); // [a1,b1,a3,b3,a5,b5,a7,b7]
            temp2 = _mm256_unpacklo_ps(rows[2], rows[3]); // [c0,d0,c2,d2,c4,d4,c6,d6]
            temp3 = _mm256_unpackhi_ps(rows[2], rows[3]); // [c1,d1,c3,d3,c5,d5,c7,d7]
            temp4 = _mm256_unpacklo_ps(rows[4], rows[5]); // [e0,f0,e2,f2,e4,f4,e6,f6]
            temp5 = _mm256_unpackhi_ps(rows[4], rows[5]); // [e1,f1,e3,f3,e5,f5,e7,f7]
            temp6 = _mm256_unpacklo_ps(rows[6], rows[7]); // [g0,h0,g2,h2,g4,h4,g6,h6]
            temp7 = _mm256_unpackhi_ps(rows[6], rows[7]); // [g1,h1,g3,h3,g5,h5,g7,h7]
            
            // Interleave 64-bit elements
            out0 = _mm256_shuffle_ps(temp0, temp2, 0x44); // [a0,b0,c0,d0,a4,b4,c4,d4]
            out1 = _mm256_shuffle_ps(temp0, temp2, 0xEE); // [a2,b2,c2,d2,a6,b6,c6,d6]
            out2 = _mm256_shuffle_ps(temp1, temp3, 0x44); // [a1,b1,c1,d1,a5,b5,c5,d5]
            out3 = _mm256_shuffle_ps(temp1, temp3, 0xEE); // [a3,b3,c3,d3,a7,b7,c7,d7]
            out4 = _mm256_shuffle_ps(temp4, temp6, 0x44); // [e0,f0,g0,h0,e4,f4,g4,h4]
            out5 = _mm256_shuffle_ps(temp4, temp6, 0xEE); // [e2,f2,g2,h2,e6,f6,g6,h6]
            out6 = _mm256_shuffle_ps(temp5, temp7, 0x44); // [e1,f1,g1,h1,e5,f5,g5,h5]
            out7 = _mm256_shuffle_ps(temp5, temp7, 0xEE); // [e3,f3,g3,h3,e7,f7,g7,h7]
            
            // Interleave 128-bit elements
            temp0 = _mm256_permute2f128_ps(out0, out4, 0x20); // [a0,b0,c0,d0,e0,f0,g0,h0]
            temp1 = _mm256_permute2f128_ps(out2, out6, 0x20); // [a1,b1,c1,d1,e1,f1,g1,h1]
            temp2 = _mm256_permute2f128_ps(out1, out5, 0x20); // [a2,b2,c2,d2,e2,f2,g2,h2]
            temp3 = _mm256_permute2f128_ps(out3, out7, 0x20); // [a3,b3,c3,d3,e3,f3,g3,h3]
            temp4 = _mm256_permute2f128_ps(out0, out4, 0x31); // [a4,b4,c4,d4,e4,f4,g4,h4]
            temp5 = _mm256_permute2f128_ps(out2, out6, 0x31); // [a5,b5,c5,d5,e5,f5,g5,h5]
            temp6 = _mm256_permute2f128_ps(out1, out5, 0x31); // [a6,b6,c6,d6,e6,f6,g6,h6]
            temp7 = _mm256_permute2f128_ps(out3, out7, 0x31); // [a7,b7,c7,d7,e7,f7,g7,h7]
            
            // Store the transposed 8x8 block
            _mm256_storeu_ps(&b->data[j * b->stride + i], temp0);
            _mm256_storeu_ps(&b->data[(j+1) * b->stride + i], temp1);
            _mm256_storeu_ps(&b->data[(j+2) * b->stride + i], temp2);
            _mm256_storeu_ps(&b->data[(j+3) * b->stride + i], temp3);
            _mm256_storeu_ps(&b->data[(j+4) * b->stride + i], temp4);
            _mm256_storeu_ps(&b->data[(j+5) * b->stride + i], temp5);
            _mm256_storeu_ps(&b->data[(j+6) * b->stride + i], temp6);
            _mm256_storeu_ps(&b->data[(j+7) * b->stride + i], temp7);
        }
    }
    
    // Handle remaining rows and columns using scalar operations
    
    // Handle remaining rows
    for (size_t i = (a->row / block_size) * block_size; i < a->row; i++) {
        for (size_t j = 0; j < a->col; j++) {
            float val = a->data[i * a->stride + j];
            b->data[j * b->stride + i] = val;
        }
    }
    
    // Handle remaining columns
    for (size_t i = 0; i < (a->row / block_size) * block_size; i++) {
        for (size_t j = (a->col / block_size) * block_size; j < a->col; j++) {
            float val = a->data[i * a->stride + j];
            b->data[j * b->stride + i] = val;
        }
    }
    
    return MATRIX_SUCCESS;
}

float matdot(const float *a, const float *b, size_t length) {
    float result = 0.0f;

    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 7 < length; i+=8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 mult = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, mult);
    }

    /* Horizontal sum of vector elements */
    float temp[8] MAT_ALIGN;
    _mm256_store_ps(temp, sum);
    result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    
    /* Handle remaining elements */
    for (; i < length; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

mat_status_t matvecmul(const mat_t *a, const float *x, float *y) {
    if (!a || !x || !y) {
        return MATRIX_NULL_POINTER;
    }
    
    for (size_t i = 0; i < a->row; ++i) {
        float *a_row = &a->data[i * a->stride];
        y[i] = matdot(a_row, x, a->col);
    }

    return MATRIX_SUCCESS;
}

mat_status_t matreshapeip(mat_t *a, size_t new_row, size_t new_col) {
    if (!a) {
        return MATRIX_NULL_POINTER;
    }
    
    // Check that the total number of elements remains the same
    if (a->row * a->col != new_row * new_col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    // Calculate new stride based on new column count
    size_t new_stride = new_col;
    if (new_col % (SIMD_ALIGN / sizeof(float)) != 0) {
        new_stride = ((new_col / (SIMD_ALIGN / sizeof(float))) + 1) * (SIMD_ALIGN / sizeof(float));
    }
    
    // If the current stride is sufficient for the new dimensions
    // (this is a common case when we're just reinterpreting the shape)
    if (new_stride <= a->stride) {
        a->row = new_row;
        a->col = new_col;
        // Keep the existing stride to maintain alignment
        return MATRIX_SUCCESS;
    }
    
    // If we get here, we need to rearrange the data to handle the new stride
    // This is more complex and would require copying the data
    // For embedded systems, it's typically better to create a new matrix
    // with the correct dimensions rather than trying to reshape in-place
    return MATRIX_UNSUPPORTED_OPERATION;
}

/**
 * @brief Reshape a matrix to new dimensions and store in a different matrix
 * 
 * This function copies data from matrix 'a' to matrix 'b' with new dimensions,
 * preserving the data in row-major order. The matrix 'b' should be allocated
 * using matalloc or matcreate before calling this function.
 * 
 * This version is optimized for embedded systems with limited memory.
 * 
 * @param a Input matrix
 * @param row New row count
 * @param col New column count
 * @param b Output matrix (must be pre-allocated with proper dimensions)
 * @return mat_status_t Status code
 */
mat_status_t matreshape(const mat_t *a, size_t row, size_t col, mat_t *b) {
    // Validate input parameters
    if (!a || !b) {
        return MATRIX_NULL_POINTER;
    }
    
    // Check that total element count matches
    if (a->row * a->col != row * col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    // Check if the destination matrix has the right dimensions
    if (b->row != row || b->col != col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    // Direct element-by-element copy with logical index tracking
    size_t total_elements = a->row * a->col;
    
    // Use direct indexing without temporary storage
    for (size_t idx = 0; idx < total_elements; idx++) {
        // Calculate source coordinates
        size_t a_row = idx / a->col;
        size_t a_col = idx % a->col;
        
        // Calculate destination coordinates
        size_t b_row = idx / col;
        size_t b_col = idx % col;
        
        // Copy the element with stride handling
        b->data[b_row * b->stride + b_col] = a->data[a_row * a->stride + a_col];
    }
    
    return MATRIX_SUCCESS;
}

/**
 * @brief Reshape with support for region allocation
 * 
 * This version of reshape creates a new matrix with the specified dimensions
 * in the provided memory region, and copies the data from the source matrix.
 * Optimized for memory-constrained embedded systems.
 * 
 * @param reg Memory region for allocation
 * @param a Input matrix
 * @param row New row count
 * @param col New column count
 * @param b Output matrix (will be allocated in the region)
 * @return mat_status_t Status code
 */
/*
mat_status_t matreshapereg(mat_region_t *reg, const mat_t MAT_IN *a, 
                               size_t row, size_t col, mat_t MAT_OUT *b) {
    if (!reg || !a || !b) {
        return MATRIX_NULL_POINTER;
    }
    
    // Check that total element count matches
    if (a->row * a->col != row * col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    // Allocate the destination matrix in the provided region
    mat_status_t status = matalloc(reg, row, col, b);
    if (status != MATRIX_SUCCESS) {
        return status;
    }
    
    // Use direct indexing without temporary storage
    size_t total_elements = a->row * a->col;
    for (size_t idx = 0; idx < total_elements; idx++) {
        // Calculate source coordinates
        size_t a_row = idx / a->col;
        size_t a_col = idx % a->col;
        
        // Calculate destination coordinates
        size_t b_row = idx / col;
        size_t b_col = idx % col;
        
        // Copy the element with stride handling
        b->data[b_row * b->stride + b_col] = a->data[a_row * a->stride + a_col];
    }
    
    return MATRIX_SUCCESS;
}
*/

/**
 * @brief Block-based matrix reshape for better memory efficiency
 * 
 * This version processes small blocks at a time to maintain cache efficiency
 * while avoiding large temporary buffers. Suitable for embedded systems.
 * 
 * @param a Input matrix
 * @param row New row count
 * @param col New column count
 * @param b Output matrix (must be pre-allocated with proper dimensions)
 * @return mat_status_t Status code
 */
mat_status_t matreshapeblock(const mat_t *a, size_t row, size_t col, mat_t *b) {
    // Validate input parameters
    if (!a || !b) {
        return MATRIX_NULL_POINTER;
    }
    
    // Check that total element count matches
    if (a->row * a->col != row * col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    // Check if the destination matrix has the right dimensions
    if (b->row != row || b->col != col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    // Small fixed-size buffer for better cache efficiency
    // 16 floats is a reasonable compromise between cache efficiency and memory usage
    // Adjust the size based on your target platform's constraints
    const size_t BLOCK_SIZE = 16;
    float block_buffer[BLOCK_SIZE];
    
    size_t total_elements = a->row * a->col;
    size_t elements_processed = 0;
    
    while (elements_processed < total_elements) {
        // Determine block size for this iteration
        size_t current_block_size = (total_elements - elements_processed < BLOCK_SIZE) ? 
                                   (total_elements - elements_processed) : BLOCK_SIZE;
        
        // Fill the buffer with a block of elements
        for (size_t i = 0; i < current_block_size; i++) {
            size_t idx = elements_processed + i;
            size_t a_row = idx / a->col;
            size_t a_col = idx % a->col;
            block_buffer[i] = a->data[a_row * a->stride + a_col];
        }
        
        // Write the buffer to the destination matrix
        for (size_t i = 0; i < current_block_size; i++) {
            size_t idx = elements_processed + i;
            size_t b_row = idx / col;
            size_t b_col = idx % col;
            b->data[b_row * b->stride + b_col] = block_buffer[i];
        }
        
        elements_processed += current_block_size;
    }
    
    return MATRIX_SUCCESS;
}

// #endif
