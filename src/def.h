#ifndef DEF_H
#define DEF_H

/*
 * Memory Alignment Explanation
 * ----------------------------
 * 
 * Memory alignment refers to the way data is arranged and accessed in computer memory.
 * The ALIGN_BOUNDARY defines how many bytes your data should be aligned to in memory.
 * 
 * AVX (32-byte alignment)
 * -----------------------
 * When using AVX instructions, data should be aligned to 32-byte boundaries because 
 * AVX registers are 256 bits (32 bytes) wide. This means the memory address where 
 * your data starts should be divisible by 32.
 * 
 * For example:
 * - Memory address 0x00000000 is aligned (0 is divisible by 32)
 * - Memory address 0x00000020 is aligned (32 is divisible by 32)
 * - Memory address 0x00000040 is aligned (64 is divisible by 32)
 * - Memory address 0x0000001F is NOT aligned (31 is not divisible by 32)
 * 
 * NEON (16-byte alignment)
 * ------------------------
 * ARM NEON instructions use 128-bit (16-byte) registers, so data should be 
 * aligned to 16-byte boundaries.
 * 
 * For example:
 * - Memory address 0x00000000 is aligned (0 is divisible by 16)
 * - Memory address 0x00000010 is aligned (16 is divisible by 16)
 * - Memory address 0x00000020 is aligned (32 is divisible by 16)
 * - Memory address 0x00000008 is NOT aligned (8 is not divisible by 16)
 * 
 * Regular float (4-byte alignment)
 * --------------------------------
 * For regular floating-point operations without SIMD, standard float 
 * alignment of 4 bytes is sufficient.
 * 
 * Why Alignment Matters - A Practical Example
 * ------------------------------------------
 * Let's say you have a matrix with 5 floats per row and you're using AVX:
 * 
 * [f1][f2][f3][f4][f5]   // Row 1
 * [f6][f7][f8][f9][f10]  // Row 2
 * 
 * Each float takes 4 bytes, so each row takes 20 bytes without padding.
 * 
 * However, with AVX, we need 32-byte alignment. To achieve this, we add padding:
 * 
 * [f1][f2][f3][f4][f5][pad][pad][pad]  // Row 1 (32 bytes with padding)
 * [f6][f7][f8][f9][f10][pad][pad][pad] // Row 2 (32 bytes with padding)
 * 
 * This is why the matrix structure includes a 'stride' variable - it accounts for 
 * this padding:
 * 
 * stride = cols;
 * if (cols % VECTOR_SIZE != 0) {
 *     stride = (cols / VECTOR_SIZE + 1) * VECTOR_SIZE;
 * }
 * 
 * When you access data, the code uses stride to calculate the correct memory address:
 * 
 * m->data[row * m->stride + col]
 * 
 * Performance Impact
 * -----------------
 * Proper alignment allows SIMD instructions to load and store data in a single 
 * operation. Misaligned data can cause:
 * 
 * 1. Performance penalties (2-10x slower on some architectures)
 * 2. Extra instructions needed to handle misaligned data
 * 3. On some processors, misaligned access can cause exceptions
 * 
 * The posix_memalign and _aligned_malloc functions used in the code ensure that 
 * memory is allocated with the proper alignment, and the ALIGNED attribute ensures 
 * that variables declared on the stack are also properly aligned.
 */

// #ifdef ENABLE_SIMD // ???
#ifdef __AVX__
#include <immintrin.h>
#define SIMD_ENABLED
#define VECTOR_SIZE 8  // AVX uses 256-bit vectors (8 floats or 4 doubles)
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_ENABLED
#define VECTOR_SIZE 4  // NEON uses 128-bit vectors (4 floats or 2 doubles)
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

// #endif // ENABLE_SIMD

/* from stdint.h */
/* Types for `void *' pointers.  */
#if __WORDSIZE == 64
# ifndef __intptr_t_defined
typedef long int		intptr_t;
#  define __intptr_t_defined
# endif
typedef unsigned long int	uintptr_t;
#else
# ifndef __intptr_t_defined
typedef int			intptr_t;
#  define __intptr_t_defined
# endif
typedef unsigned int		uintptr_t;
#endif

/*
Add GPU acceleration support since Rasberry pi 5 can be modified to work with GPUs 
*/

typedef struct {
    size_t row;
    size_t col;
    size_t stride; /* Stride between rows (for padding) */
    float *data;   /* keep in mind: NVIDIA GPUs with tensor cores and
                      many deep learning models can tolerate reduced precision (float16) */
} mat_t;

#define __INPUT  
#define __OUTPUT  
#define __NULLABLE

typedef __SIZE_TYPE__ size_t;
typedef unsigned char mat_status_t;

/* Matrix status list */
#define MATRIX_SUCCESS              ((mat_status_t)0)
#define MATRIX_NOT_INITIALIZED      ((mat_status_t)1)
#define MATRIX_NULL_POINTER         ((mat_status_t)2)
#define MATRIX_REGION_FULL          ((mat_status_t)3)
#define MATRIX_INVALID_REGION       ((mat_status_t)4)
#define MATRIX_DIMENSION_MISMATCH   ((mat_status_t)5)
// ...

#endif // DEF_H
