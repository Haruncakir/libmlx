// (AVX) gcc -O3 -fopenmp -mavx -Wall matrix_test.c -o simd_matrix_example
// (NEON) gcc -O3 -fopenmp -mfpu=neon -Wall matrix_test.c -o simd_matrix_example
// -march=native: Optimizes for your specific CPU (e.g., uses AVX2 if available)

#include "matpool.h"
#include <stdio.h>

#define REGION_SIZE (1024 * 4) // 4KB

// #define safe(func) if ((func) != MATRIX_SUCCESS) { return 1; }
// gcc -O3 -Wall  matrix_test.c matrix.c matpool.c -o matrix_example

int main() {
    static unsigned char memory_pool[REGION_SIZE];

    mat_region_t reg;
    mat_status_t stat = reginit(&reg, memory_pool, REGION_SIZE);

    if (stat != MATRIX_SUCCESS)
        return 1;
    
    mat_t A, B, C;
    float A_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};     /* 2x3 matrix */
    float B_data[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};  /* 3x2 matrix */

    if (matcreate(&reg, 2, 3, A_data, &A) != MATRIX_SUCCESS)
        return 2;
    
    if (matcreate(&reg, 3, 2, B_data, &B) != MATRIX_SUCCESS)
        return 3;
    
    if (matresalloc(&reg, &A, &B, &C) != MATRIX_SUCCESS)
        return 4;
    
    if (matmul(&A, &B, &C) != MATRIX_SUCCESS)
        return 5;
    
     /* No need to free individual matrices - just reset the region when done */
    printf("\nRegion statistics before reset:\n");
    printf("Total size: %zu bytes\n", reg.size);
    printf("Used: %zu bytes\n", reg.used);
    printf("Matrices allocated: %zu\n", reg.mat_count);
    
    /* Reset region for reuse */
    regreset(&reg);
    
    printf("\nRegion statistics after reset:\n");
    printf("Total size: %zu bytes\n", reg.size);
    printf("Used: %zu bytes\n", reg.used);
    printf("Matrices allocated: %zu\n", reg.mat_count);

    return 0;
}
