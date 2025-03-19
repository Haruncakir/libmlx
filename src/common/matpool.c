#include "../../include/matrix/matpool.h"
#include "../../include/matrix/matrix_config.h"

/* 
 * Utility function to align memory addresses to SIMD boundary
 * This ensures proper alignment for AVX (32-byte), NEON (16-byte),
 * or standard float (4-byte) operations depending on architecture.
 */
static size_t align_size(size_t size) {
    return (size + (SIMD_ALIGN - 1)) & ~(SIMD_ALIGN - 1);
}

/* 
 * Utility function to check and align a pointer address
 * Returns the next aligned address from the given pointer
 */
static unsigned char* align_ptr(unsigned char* ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned = (addr + (SIMD_ALIGN - 1)) & ~(SIMD_ALIGN - 1);
    return (unsigned char*)aligned;
}

mat_status_t reginit(mat_region_t *reg, void *memory, size_t size) {
    if (!reg || !memory || size == 0)
        return MATRIX_NULL_POINTER;
    
    /* Ensure the starting address is aligned */
    unsigned char *aligned_memory = align_ptr((unsigned char*)memory);
    
    /* Adjust size to account for alignment of the starting pointer */
    size_t adjustment = aligned_memory - (unsigned char*)memory;
    if (adjustment >= size) {
        return MATRIX_INVALID_REGION; /* Region too small after alignment */
    }
    
    reg->memory = aligned_memory;
    reg->size = size - adjustment;
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

mat_status_t matalloc(mat_region_t *reg, size_t rows, size_t cols, mat_t MAT_OUT *mat) {
    if (!reg || !mat)
        return MATRIX_NULL_POINTER;
    if (rows == 0 || cols == 0)
        return MATRIX_DIMENSION_MISMATCH;
   
    /* Calculate matrix stride for proper SIMD alignment */
    //size_t stride = cols;
    //if (cols % (SIMD_ALIGN / sizeof(float)) != 0) {
    //    stride = ((cols / (SIMD_ALIGN / sizeof(float))) + 1) * (SIMD_ALIGN / sizeof(float));
    //}
    size_t floats_per_align = SIMD_ALIGN / sizeof(float);
    size_t stride = ((cols + floats_per_align - 1) / floats_per_align) * floats_per_align;
    
    /* Allocate memory for matrix with stride for proper row alignment */
    size_t data_size = rows * stride * sizeof(float);
    float *data = (float*)regalloc(reg, data_size);
    if (!data)
        return MATRIX_REGION_FULL;
    
    for (size_t i = 0; i < rows * stride; i++) {
        data[i] = 0.0f;  // Initialize all memory (including padding due to stride)
    }
   
    mat->row = rows;
    mat->col = cols;
    mat->stride = stride;
    mat->data = data;
    reg->mat_count++;
    
    return MATRIX_SUCCESS;
}

mat_status_t matcreate(mat_region_t *reg,
        size_t rows, size_t cols, const float *data,
        mat_t *mat) {
    mat_status_t stat = matalloc(reg, rows, cols, mat);
    if (stat != MATRIX_SUCCESS)
        return stat;
   
    if (data) {
        /* Copy data with stride handling for proper SIMD alignment */
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                mat->data[i * mat->stride + j] = data[i * cols + j];
            }
        }
    }
    
    return MATRIX_SUCCESS;
}

mat_status_t matresalloc(mat_region_t *reg, const mat_t MAT_IN *A,
                        const mat_t MAT_IN *B, mat_t MAT_OUT *C) {
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
