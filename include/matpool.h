#ifndef MATPOOL_H
#define MATPOOL_H

#include "def.h"

/*
 * Memory Alignment for SIMD Operations
 * ------------------------------------
 * 
 * For SIMD operations, memory must be aligned to specific boundaries:
 * - AVX: 32-byte alignment (256-bit registers)
 * - NEON: 16-byte alignment (128-bit registers)
 * - Default: 4-byte alignment (regular float operations)
 * 
 * Proper alignment enables:
 * 1. Single-cycle memory operations with SIMD instructions
 * 2. Prevention of performance penalties (2-10x slowdown with misaligned access)
 * 3. Avoidance of potential crashes on some architectures that require alignment
 * 
 * The pool implementation uses aligned_size() to ensure all allocations
 * are properly aligned for the target architecture's SIMD instructions.
 */


/*
 * Region-based memory management
 * For matrix operations with predictable sizes and lifetimes,
 * region-based allocation is particularly well-suited and
 * can significantly outperform malloc both in performance and safety.
 */
typedef struct {
    unsigned char *memory;    /* Base pointer to the region's memory */
    size_t size;              /* Total size of the region in bytes */
    size_t used;              /* Currently used bytes in the region */
    size_t mat_count;         /* Number of matrices allocated in this region */
} mat_region_t;

mat_status_t reginit(mat_region_t *reg, void *memory, size_t size);
mat_status_t regreset(mat_region_t *reg);
mat_status_t matalloc(mat_region_t *reg, size_t rows, size_t cols, mat_t __OUTPUT *mat);
mat_status_t matcreate(mat_region_t *reg, size_t rows, size_t cols, 
                        const float __NULLABLE *data, mat_t *mat);
mat_status_t matresalloc(mat_region_t *reg, const mat_t __INPUT *A,
                        const mat_t __INPUT *B, mat_t __OUTPUT *C);
const char *strmaterr(mat_status_t stat);

#endif // MATPOOL_H
