/**
 * Author - Harun Cakir
 * CONSIDER: Maybe have a macro like AUTHOR() ...
 * 
 * matrix.h - SIMD-optimized matrix operations using both AVX and NEON
 * 
 * This header provides a Matrix data structure and operations optimized for
 * SIMD instructions on both x86 (AVX) and ARM (NEON) architectures.
 * Added OpenMP parallelization for multi-core systems.
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

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>

#define POS_INF (1.0/0.0)
#define NEG_INF (-1.0/0.0)

#define LOCAL_DEV
#ifdef LOCAL_DEV
#include <immintrin.h>
// #include <arm_neon.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#define SIMD_ENABLED
#define VECTOR_SIZE 8  // AVX uses 256-bit vectors (8 floats)
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_ENABLED
#define VECTOR_SIZE 4  // NEON uses 128-bit vectors (4 floats)
#else
#define VECTOR_SIZE 1  // Fallback to scalar operations
#endif

// Align to SIMD vector boundary
#ifdef __AVX__
#define ALIGN_BOUNDARY 32  // AVX 256-bit alignment
#elif defined(__ARM_NEON)
#define ALIGN_BOUNDARY 16  // NEON 128-bit alignment
#else
#define ALIGN_BOUNDARY 4   // Regular float alignment
#endif

// Alignment macro
#ifdef _MSC_VER
#define ALIGNED __declspec(align(ALIGN_BOUNDARY))
#else
#define ALIGNED __attribute__((aligned(ALIGN_BOUNDARY)))
#endif

/**
 * will work with caller-provided buffers for zero-copy execution  
 */
typedef struct {
    float* data;           // Pointer to matrix data (caller provided)
    size_t rows;           // Number of rows
    size_t cols;           // Number of columns
    size_t stride;         // Stride between rows (for padding)
    bool owns_memory;      // Flag to indicate if we should free memory
} Matrix;

// Matrix view - allows working with submatrices without copying
typedef struct {
    Matrix base;           // Base matrix info
    size_t row_offset;     // Starting row
    size_t col_offset;     // Starting column
    size_t num_rows;       // Number of rows in view
    size_t num_cols;       // Number of columns in view
} MatrixView;

// Initialize a matrix with caller-provided buffer
Matrix matrix_create_with_buffer(float* buffer, size_t rows, size_t cols, size_t stride) {
    Matrix m;
    m.data = buffer;
    m.rows = rows;
    m.cols = cols;
    m.stride = stride == 0 ? cols : stride;  // If stride is 0, use cols as stride
    m.owns_memory = false;
    return m;
}

// Allocate a new aligned matrix and return it
Matrix matrix_create(size_t rows, size_t cols) {
    // Calculate stride with padding for alignment
    size_t stride = cols;
    if (cols % VECTOR_SIZE != 0) {
        stride = (cols / VECTOR_SIZE + 1) * VECTOR_SIZE;
    }
    
    // Allocate aligned memory
    float* data = NULL;
#ifdef _MSC_VER
    data = (float*)_aligned_malloc(rows * stride * sizeof(float), ALIGN_BOUNDARY);
#else
    int result = posix_memalign((void**)&data, ALIGN_BOUNDARY, rows * stride * sizeof(float));
    if (result != 0) {
        data = NULL;
    }
#endif

    Matrix m;
    m.data = data;
    m.rows = rows;
    m.cols = cols;
    m.stride = stride;
    m.owns_memory = true;
    
    return m;
}

// Free matrix if it owns its memory
void matrix_destroy(Matrix* m) {
    if (m->owns_memory && m->data != NULL) {
#ifdef _MSC_VER
        _aligned_free(m->data);
#else
        free(m->data);
#endif
        m->data = NULL;
    }
}

// Get element at (row, col)
float matrix_get(const Matrix* m, size_t row, size_t col) {
    return m->data[row * m->stride + col];
}

// Set element at (row, col)
void matrix_set(Matrix* m, size_t row, size_t col, float value) {
    m->data[row * m->stride + col] = value;
}

// Create a matrix view from an existing matrix
MatrixView matrix_view_create(const Matrix* base, size_t row_offset, size_t col_offset, 
                             size_t num_rows, size_t num_cols) {
    MatrixView view;
    view.base = *base;  // Copy the base matrix info
    view.row_offset = row_offset;
    view.col_offset = col_offset;
    view.num_rows = num_rows;
    view.num_cols = num_cols;
    return view;
}

// Get element from a matrix view
float matrix_view_get(const MatrixView* view, size_t row, size_t col) {
    size_t actual_row = view->row_offset + row;
    size_t actual_col = view->col_offset + col;
    return view->base.data[actual_row * view->base.stride + actual_col];
}

// Set element in a matrix view
void matrix_view_set(MatrixView* view, size_t row, size_t col, float value) {
    size_t actual_row = view->row_offset + row;
    size_t actual_col = view->col_offset + col;
    view->base.data[actual_row * view->base.stride + actual_col] = value;
}

// Fill matrix with a value
void matrix_fill(Matrix* m, float value) {
    #pragma omp parallel for
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m->data[i * m->stride + j] = value;
        }
    }
}

// Copy matrix src to dst (assuming same dimensions)
bool matrix_copy(Matrix* dst, const Matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        return false;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dst->data[i * dst->stride + j] = src->data[i * src->stride + j];
        }
    }
    
    return true;
}

// Get pointer to start of row for direct SIMD operations
float* matrix_row_ptr(Matrix* m, size_t row) {
    return &m->data[row * m->stride];
}

// SIMD-optimized matrix addition: C = A + B
bool matrix_add(Matrix* c, const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols || 
        c->rows != a->rows || c->cols != a->cols) {
        return false;
    }

#ifdef SIMD_ENABLED
    #pragma omp parallel for
    for (size_t i = 0; i < a->rows; i++) {
        size_t j = 0;
        float* a_row = &a->data[i * a->stride];
        float* b_row = &b->data[i * b->stride];
        float* c_row = &c->data[i * c->stride];

#ifdef __AVX__
        // Process 8 elements at a time with AVX
        for (; j + 7 < a->cols; j += 8) {
            __m256 va = _mm256_loadu_ps(&a_row[j]);
            __m256 vb = _mm256_loadu_ps(&b_row[j]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&c_row[j], vc);
        }
#elif defined(__ARM_NEON)
        // Process 4 elements at a time with NEON
        for (; j + 3 < a->cols; j += 4) {
            float32x4_t va = vld1q_f32(&a_row[j]);
            float32x4_t vb = vld1q_f32(&b_row[j]);
            float32x4_t vc = vaddq_f32(va, vb);
            vst1q_f32(&c_row[j], vc);
        }
#endif

        // Handle remaining elements
        for (; j < a->cols; j++) {
            c_row[j] = a_row[j] + b_row[j];
        }
    }
#else
    // Fallback for non-SIMD systems
    #pragma omp parallel for
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            matrix_set(c, i, j, matrix_get(a, i, j) + matrix_get(b, i, j));
        }
    }
#endif

    return true;
}

// SIMD-optimized dot product of two vectors
float vector_dot_product(const float* a, const float* b, size_t length) {
    float result = 0.0f;
    
#ifdef __AVX__
    // AVX implementation for dot product
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 mult = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, mult);
    }
    
    // Horizontal sum of vector elements
    float temp[8] ALIGNED;
    _mm256_store_ps(temp, sum);
    result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    
    // Handle remaining elements
    for (; i < length; i++) {
        result += a[i] * b[i];
    }
#elif defined(__ARM_NEON)
    // NEON implementation for dot product
    float32x4_t sum = vdupq_n_f32(0);
    size_t i = 0;
    
    for (; i + 3 < length; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        sum = vmlaq_f32(sum, va, vb);  // Multiply and accumulate
    }
    
    // Horizontal sum of vector elements
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum2 = vpadd_f32(sum2, sum2);
    result = vget_lane_f32(sum2, 0);
    
    // Handle remaining elements
    for (; i < length; i++) {
        result += a[i] * b[i];
    }
#else
    // Scalar implementation for dot product
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
#endif

    return result;
}

// Compute dot product of two matrix rows
float matrix_row_dot_product(const Matrix* a, size_t row_a, const Matrix* b, size_t row_b) {
    if (a->cols != b->cols) {
        return 0.0f;  // Incompatible dimensions
    }
    
    const float* a_row = &a->data[row_a * a->stride];
    const float* b_row = &b->data[row_b * b->stride];
    
    return vector_dot_product(a_row, b_row, a->cols);
}

// SIMD-optimized matrix-vector multiplication: y = A * x
bool matrix_vector_multiply(float* y, const Matrix* a, const float* x) {
    // For small matrices, the overhead of parallel might not be worth it
    // For large matrices, parallelizing the outer loop gives good speedup
    if (a->rows >= 64) {  // Only parallelize for larger matrices
        #pragma omp parallel for
        for (size_t i = 0; i < a->rows; i++) {
            float* a_row = &a->data[i * a->stride];
            y[i] = vector_dot_product(a_row, x, a->cols);
        }
    } else {
        for (size_t i = 0; i < a->rows; i++) {
            float* a_row = &a->data[i * a->stride];
            y[i] = vector_dot_product(a_row, x, a->cols);
        }
    }
    
    return true;
}

// SIMD-optimized matrix multiplication: C = A * B
bool matrix_multiply(Matrix* c, const Matrix* a, const Matrix* b) {
    if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols) {
        return false;
    }

    // Initialize C to zeros
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < c->rows; i++) {
        for (size_t j = 0; j < c->cols; j++) {
            c->data[i * c->stride + j] = 0.0f;
        }
    }

#ifdef __AVX__
    // AVX implementation for matrix multiplication
    #pragma omp parallel for
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t k = 0; k < a->cols; k++) {
            __m256 va = _mm256_set1_ps(matrix_get(a, i, k));
            size_t j = 0;
            
            for (; j + 7 < b->cols; j += 8) {
                float* c_row = matrix_row_ptr(c, i);
                float* b_row = &b->data[k * b->stride];
                
                __m256 vb = _mm256_loadu_ps(&b_row[j]);
                __m256 vc = _mm256_loadu_ps(&c_row[j]);
                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                _mm256_storeu_ps(&c_row[j], vc);
            }
            
            // Handle remaining columns
            for (; j < b->cols; j++) {
                float c_val = matrix_get(c, i, j);
                float a_val = matrix_get(a, i, k);
                float b_val = matrix_get(b, k, j);
                matrix_set(c, i, j, c_val + a_val * b_val);
            }
        }
    }
#elif defined(__ARM_NEON)
    // NEON implementation for matrix multiplication
    #pragma omp parallel for
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t k = 0; k < a->cols; k++) {
            float32x4_t va = vdupq_n_f32(matrix_get(a, i, k));
            size_t j = 0;
            
            for (; j + 3 < b->cols; j += 4) {
                float* c_row = matrix_row_ptr(c, i);
                float* b_row = &b->data[k * b->stride];
                
                float32x4_t vb = vld1q_f32(&b_row[j]);
                float32x4_t vc = vld1q_f32(&c_row[j]);
                vc = vaddq_f32(vc, vmulq_f32(va, vb));
                vst1q_f32(&c_row[j], vc);
            }
            
            // Handle remaining columns
            for (; j < b->cols; j++) {
                float c_val = matrix_get(c, i, j);
                float a_val = matrix_get(a, i, k);
                float b_val = matrix_get(b, k, j);
                matrix_set(c, i, j, c_val + a_val * b_val);
            }
        }
    }
#else
    // Scalar implementation for matrix multiplication with better parallelization
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++) {
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(c, i, j, sum);
        }
    }
#endif

    return true;
}

// Parallel implementation of matrix transpose: B = A^T
bool matrix_transpose(Matrix* b, const Matrix* a) {
    if (b->rows != a->cols || b->cols != a->rows) {
        return false;
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            matrix_set(b, j, i, matrix_get(a, i, j));
        }
    }
    
    return true;
}

// Parallel reduction for finding matrix maximum value
float matrix_max(const Matrix* m) {
    float max_val = matrix_get(m, 0, 0);
    
    #pragma omp parallel
    {
        float local_max = NEG_INF;
        
        #pragma omp for collapse(2) nowait
        for (size_t i = 0; i < m->rows; i++) {
            for (size_t j = 0; j < m->cols; j++) {
                float val = matrix_get(m, i, j);
                if (val > local_max) {
                    local_max = val;
                }
            }
        }
        
        #pragma omp critical
        {
            if (local_max > max_val) {
                max_val = local_max;
            }
        }
    }
    
    return max_val;
}

// Parallel reduction for finding matrix minimum value
float matrix_min(const Matrix* m) {
    float min_val = matrix_get(m, 0, 0);
    
    #pragma omp parallel
    {
        float local_min = POS_INF;
        
        #pragma omp for collapse(2) nowait
        for (size_t i = 0; i < m->rows; i++) {
            for (size_t j = 0; j < m->cols; j++) {
                float val = matrix_get(m, i, j);
                if (val < local_min) {
                    local_min = val;
                }
            }
        }
        
        #pragma omp critical
        {
            if (local_min < min_val) {
                min_val = local_min;
            }
        }
    }
    
    return min_val;
}

// Print matrix for debugging
void matrix_print(const Matrix* m) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.4f ", matrix_get(m, i, j));
        }
        printf("\n");
    }
}

#endif // MATRIX_H