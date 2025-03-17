#ifndef MATRIX_CONFIG_H
#define MATRIX_CONFIG_H

/*
 * Memory Alignment 
 * -----------------
 * 
 * Memory alignment refers to the way data is arranged and accessed in computer memory.
 * The SIMD_ALIGN defines how many bytes your data should be aligned to in memory.
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

#if defined(__AVX__) || defined(__AVX2__)
    #define MAT_USE_AVX 1
    #define SIMD_ALIGN 32
    #define FLOAT_PER_VECTOR 8  /* 8 floats in 256-bit AVX register */
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define MAT_USE_NEON 1
    #define SIMD_ALIGN 16
    #define FLOAT_PER_VECTOR 4  /* 4 floats in 128-bit NEON register */
#else
    #define MAT_USE_SCALAR 1
    #define SIMD_ALIGN 4
    #define FLOAT_PER_VECTOR 1
#endif

/* manual overrides */
#ifdef MAT_FORCE_SCALAR
    #undef MAT_USE_AVX
    #undef MAT_USE_NEON
    #define MAT_USE_SCALAR 1
    #define SIMD_ALIGN 4
    #define FLOAT_PER_VECTOR 1
#endif

/* platform-specific alignment macros */
#ifdef _MSC_VER
    #define MAT_ALIGN __declspec(align(SIMD_ALIGN))
#else
    #define MAT_ALIGN __attribute__((aligned(SIMD_ALIGN)))
#endif

/* function attributes */
#define MAT_INLINE static inline
#ifdef __GNUC__ /* to inform the compiler about the expected outcome of a condition */
    #define MAT_LIKELY(x) __builtin_expect(!!(x), 1)
    #define MAT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define MAT_LIKELY(x) (x)
    #define MAT_UNLIKELY(x) (x)
#endif

// Define uintptr_t based on the architecture
#if defined(__SIZEOF_POINTER__) // Check if the macro is defined
    #if __SIZEOF_POINTER__ == 8
        typedef unsigned long uintptr_t; // 64-bit architecture
    #elif __SIZEOF_POINTER__ == 4
        typedef unsigned int uintptr_t; // 32-bit architecture
    #else
        #error "Unsupported pointer size"
    #endif
#else
    // Fallback definition (assuming 32-bit if not defined)
    typedef unsigned int uintptr_t; // Default to 32-bit
#endif

/* Threading model - default to single-threaded for embedded */
/*! not implemented */
#ifndef MAT_MULTI_THREADED
    #define MAT_MULTI_THREADED 0
#endif

/* memory allocation strategy */
/*! not implemented */
#ifndef MAT_USE_MALLOC
    #define MAT_USE_MALLOC 0  /* Default to not using dynamic allocation */
#endif

#endif // MATRIX_CONFIG_H
