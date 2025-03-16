// Example usage of the matrix memory pool

#include "../../include/matrix/matpool.h"
#include "../../include/matrix/matrix_config.h"

int main() {
    // Create a memory region
    unsigned char memory[1024];
    mat_region_t region;
    mat_status_t status;
    
    // Initialize the region
    status = reginit(&region, memory, sizeof(memory));
    if (status != MATRIX_SUCCESS) {
        printf("Error: %s\n", strmaterr(status));
        return -1;
    }
    
    /* Memory region visualization:
     * Before alignment:
     * [memory start][-----------------1024 bytes----------------][memory end]
     * 
     * After alignment (assuming AVX 32-byte SIMD alignment):
     * [memory start][padding][--------aligned region-----------][memory end]
     *               ^
     *               region.memory points here
     */
    
    // Allocate a 3x4 matrix
    mat_t A;
    status = matalloc(&region, 3, 4, &A);
    // A.data[13] = 0;
    
    /* Matrix A memory layout with SIMD alignment:
     * Assuming SIMD_ALIGN = 32 bytes (AVX) and sizeof(float) = 4 bytes
     * Each row needs to be aligned to 8 floats (32 bytes)
     * 
     * Physical layout (with stride):
     * [a00 a01 a02 a03 pad pad pad pad] <- Row 0: 4 values + 4 padding elements for 8-float alignment
     * [a10 a11 a12 a13 pad pad pad pad] <- Row 1: 4 values + 4 padding elements for 8-float alignment
     * [a20 a21 a22 a23 pad pad pad pad] <- Row 2: 4 values + 4 padding elements for 8-float alignment
     * 
     * Logical view (3x4 matrix):
     * [a00 a01 a02 a03]
     * [a10 a11 a12 a13]
     * [a20 a21 a22 a23]
     */
    
    // Initialize matrix A with values
    float data_A[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    
    // Create a new matrix with initial data
    mat_t B;
    status = matcreate(&region, 3, 4, data_A, &B);
    
    /* Matrix B after creation:
     * Logical view (what we work with):
     * [1.0  2.0  3.0  4.0]
     * [5.0  6.0  7.0  8.0]
     * [9.0 10.0 11.0 12.0]
     * 
     * Physical memory (what's actually stored with padding):
     * [1.0  2.0  3.0  4.0 pad pad pad pad] <- Row 0
     * [5.0  6.0  7.0  8.0 pad pad pad pad] <- Row 1
     * [9.0 10.0 11.0 12.0 pad pad pad pad] <- Row 2
     * 
     * When accessing B.data[i * B.stride + j]:
     * - i selects the row
     * - B.stride (8 in this example) ensures we skip over padding
     * - j selects the column
     */
    
    // Create a 4x2 matrix for later multiplication
    mat_t C;
    float data_C[] = {
        0.5, 1.5,
        2.5, 3.5,
        4.5, 5.5,
        6.5, 7.5
    };
    status = matcreate(&region, 4, 2, data_C, &C);
    
    /* Matrix C (4x2):
     * Logical view:
     * [0.5 1.5]
     * [2.5 3.5]
     * [4.5 5.5]
     * [6.5 7.5]
     * 
     * Physical memory (with alignment):
     * [0.5 1.5 pad pad pad pad pad pad] <- Row 0
     * [2.5 3.5 pad pad pad pad pad pad] <- Row 1
     * [4.5 5.5 pad pad pad pad pad pad] <- Row 2
     * [6.5 7.5 pad pad pad pad pad pad] <- Row 3
     */
    
    // Allocate a result matrix for multiplication B * C
    mat_t D;
    status = matresalloc(&region, &B, &C, &D);
    
    /* Matrix multiplication: B(3x4) * C(4x2) = D(3x2)
     * 
     * Dimension check: B.col(4) == C.row(4) ✓
     * Result dimensions: D.row = B.row(3), D.col = C.col(2)
     * 
     * Logical calculation:
     * D[0,0] = B[0,0]*C[0,0] + B[0,1]*C[1,0] + B[0,2]*C[2,0] + B[0,3]*C[3,0]
     *        = 1.0*0.5 + 2.0*2.5 + 3.0*4.5 + 4.0*6.5
     *        = 0.5 + 5.0 + 13.5 + 26.0
     *        = 45.0
     * 
     * Final result matrix D would be:
     * [45.0  51.0]
     * [101.0 115.0]
     * [157.0 179.0]
     */
    
    // Reset the region to free all matrices
    status = regreset(&region);
    
    /* After reset:
     * - All matrix memory is considered freed
     * - region.used = 0
     * - region.mat_count = 0
     * - The underlying memory is not modified, just marked as available for reuse
     */
    
    return 0;
}