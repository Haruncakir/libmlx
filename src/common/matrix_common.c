#include "matrix_config.h"
#include "matrix.h"

float matget(const mat_t *m, size_t row, size_t col) {
    return m->data[row * m->stride + col];
}

void matset(mat_t *m, size_t row, size_t col, float value) {
    m->data[row * m->stride + col] = value;
}

mat_status_t matfill(mat_t *m, float value) {
    for (size_t i = 0; i < m->row; ++i) {
        for (size_t j = 0; j < m->col; ++j)
            m->data[i * m->row + j] = value;
    }
}
