#include "../../include/matrix/matpool.h"
#include "../../include/matrix/matrix_config.h"

float matget(const mat_t *m, size_t row, size_t col) {
    if (!m || row >= m->row || col >= m->col) {
        return 0.0f;  /* Safe default on error */
    }
    return m->data[row * m->stride + col];
}

void matset(mat_t *m, size_t row, size_t col, float value) {
    if (!m || row >= m->row || col >= m->col) {
        return;  /* Ignore if out of bounds */
    }
    m->data[row * m->stride + col] = value;
}

mat_status_t matfill(mat_t *m, float value) {
    for (size_t i = 0; i < m->row; ++i) {
        for (size_t j = 0; j < m->col; ++j)
            m->data[i * m->row + j] = value;
    }
}

/* Get pointer to start of row for direct access */
float* matrowptr(mat_t *m, size_t row) {
    if (!m || row >= m->row) {
        return (void*)0;
    }
    return &m->data[row * m->stride];
}

/* Fill matrix with a value */
mat_status_t matfill(mat_t *m, float value) {
    if (!m) {
        return MATRIX_NULL_POINTER;
    }
    
    for (size_t i = 0; i < m->row; i++) {
        for (size_t j = 0; j < m->col; j++) {
            matset(m, i, j, value);
        }
    }
    return MATRIX_SUCCESS;
}

/* Print matrix for debugging */
/*
void matprint(const mat_t *m) {
    if (!m) {
        printf("NULL matrix\n");
        return;
    }
    
    printf("Matrix %zux%zu (stride=%zu):\n", m->row, m->col, m->stride);
    for (size_t i = 0; i < m->row; i++) {
        for (size_t j = 0; j < m->col; j++) {
            printf("%8.4f ", mat_get(m, i, j));
        }
        printf("\n");
    }
}
*/

/* Copy matrix src to dst */
mat_status_t matcopy(const mat_t *src, mat_t *dst) {
    if (!src || !dst) {
        return MATRIX_NULL_POINTER;
    }
    
    if (dst->row != src->row || dst->col != src->col) {
        return MATRIX_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < src->row; i++) {
        for (size_t j = 0; j < src->col; j++) {
            mat_set(dst, i, j, mat_get(src, i, j));
        }
    }
    
    return MATRIX_SUCCESS;
}

/* Create identity matrix */
mat_status_t matidentity(mat_t *m) {
    if (!m) {
        return MATRIX_NULL_POINTER;
    }
    
    /* Set all elements to zero first */
    mat_fill(m, 0.0f);
    
    /* Set diagonal elements to 1.0 */
    size_t min_dim = (m->row < m->col) ? m->row : m->col;
    for (size_t i = 0; i < min_dim; i++) {
        mat_set(m, i, i, 1.0f);
    }
    
    return MATRIX_SUCCESS;
}

/* Get the Frobenius norm of a matrix */
float matnorm(const mat_t *m) {
    if (!m) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < m->row; i++) {
        for (size_t j = 0; j < m->col; j++) {
            float val = mat_get(m, i, j);
            sum += val * val;
        }
    }
    
    /* Return square root of sum of squares */
    return sqrtf(sum);
}