#include "matpool.h"

/* Utility function to align memory addresses */
static size_t align_size(size_t size) {
    /* Align to 8-byte boundary for better memory access on most platforms */
    return (size + 7) & ~7;
}
mat_status_t reginit(mat_region_t *reg, void *memory, size_t size) {
    if (!reg | !memory | size == 0)
        return MATRIX_NULL_POINTER;
    reg->memory = (unsigned char *)memory;
    reg->size = size;
    reg->used = 0;
    reg->mat_count = 0;

    return MATRIX_SUCCESS;
}

mat_status_t regreset(mat_region_t *reg) {
    if (!reg)
        return MATRIX_NULL_POINTER;

    reg->used = 0;
    reg->mat_count = 0;

    return MATRIX_SUCCESS;
}

static void *regalloc(mat_region_t *reg, size_t size) {
    size_t aligned = align_size(size);

    if (reg->used + aligned > reg->size)
        return (void*)0;
    void *ptr = reg->memory + reg->used;
    reg->used += aligned;

    return ptr;
}

mat_status_t matalloc(mat_region_t *reg, size_t rows, size_t cols, mat_t __OUTPUT *mat) {
    if (!reg | !mat)
        return MATRIX_NULL_POINTER;
    if (rows == 0 || cols == 0)
        return MATRIX_DIMENSION_MISMATCH;
    
    size_t data_size = rows * cols * sizeof(float);
    float *data = (float*)regalloc(reg, data_size);

    if (!data)
        return MATRIX_REGION_FULL;
    
    mat->row = rows;
    mat->col = cols;
    mat->data = data;

    reg->mat_count++;

    return MATRIX_SUCCESS;
}

mat_status_t matcreate(mat_region_t *reg, 
        size_t rows, size_t cols, const float __NULLABLE *data,
        mat_t *mat) {
    mat_status_t stat = matalloc(reg, rows, cols, mat);
    if (stat != MATRIX_SUCCESS)
        return stat;
    
    if (data) {
        size_t total_elems = rows * cols;
        size_t i = 0;
        for (; i < total_elems; ++i)
            mat->data[i] = data[i];
    }
}

mat_status_t matresalloc(mat_region_t *reg, const mat_t __INPUT *A,
                        const mat_t __INPUT *B, const mat_t __OUTPUT *C) {
    if (!A || !B || !C)
        return MATRIX_NULL_POINTER;
    
    if (A->col != B->row)
        return MATRIX_DIMENSION_MISMATCH;
    
    return matalloc(reg, A->row, B->col, C);
}

const char *strmaterr(mat_status_t stat) {
    switch(stat) {
        case MATRIX_SUCCESS:
            return "Success";
        case MATRIX_REGION_FULL:
            return "Memory region is full";
        case MATRIX_DIMENSION_MISMATCH:
            return "Matrix dimension mismatch";
        case MATRIX_INVALID_REGION:
            return "Invalid memory region";
        case MATRIX_NULL_POINTER:
            return "Null pointer provided";
        default:
            return "Unknown matrix error";
    }
}
