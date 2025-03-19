#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "../../include/matrix/matpool.h"
#include "../../include/matrix/matrix_config.h"

#define TEST_PASSED "\033[32mPASSED\033[0m"
#define TEST_FAILED "\033[31mFAILED\033[0m"

// Define buffer sizes for memory regions
#define SMALL_REGION_SIZE (1024)
#define MEDIUM_REGION_SIZE (16 * 1024)
#define LARGE_REGION_SIZE (256 * 1024)

// Helper function to check if a pointer is aligned to SIMD_ALIGN
static int is_aligned(const void* ptr) {
    return ((uintptr_t)ptr & (SIMD_ALIGN - 1)) == 0;
}

// Test region initialization
const char* test_reginit() {
    unsigned char* memory = (unsigned char*)malloc(SMALL_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    // Intentionally misalign the memory to test alignment correction
    unsigned char* misaligned_memory = memory + 1; // +1 ensures misalignment
    
    mat_region_t region;
    mat_status_t status = reginit(&region, misaligned_memory, SMALL_REGION_SIZE - 1);
    
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "reginit failed to initialize region";
    }
    
    // Check if memory was properly aligned
    if (!is_aligned(region.memory)) {
        free(memory);
        return "reginit did not properly align memory";
    }
    
    // Check if size was adjusted correctly
    if (region.size >= SMALL_REGION_SIZE - 1) {
        free(memory);
        return "reginit did not adjust size correctly for alignment";
    }
    
    // Check if used space and matrix count are initialized to zero
    if (region.used != 0 || region.mat_count != 0) {
        free(memory);
        return "reginit did not initialize used space and matrix count to zero";
    }
    
    // Test with NULL parameters
    if (reginit(NULL, memory, SMALL_REGION_SIZE) != MATRIX_NULL_POINTER) {
        free(memory);
        return "reginit failed to detect NULL region pointer";
    }
    
    if (reginit(&region, NULL, SMALL_REGION_SIZE) != MATRIX_NULL_POINTER) {
        free(memory);
        return "reginit failed to detect NULL memory pointer";
    }
    
    if (reginit(&region, memory, 0) != MATRIX_NULL_POINTER) {
        free(memory);
        return "reginit failed to detect zero size";
    }
    
    // Test with too small region after alignment
    unsigned char tiny_memory[SIMD_ALIGN + 1];
    status = reginit(&region, tiny_memory + 1, SIMD_ALIGN);
    if (status != MATRIX_INVALID_REGION) {
        free(memory);
        return "reginit failed to detect region too small after alignment";
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test region reset
const char* test_regreset() {
    unsigned char* memory = (unsigned char*)malloc(SMALL_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, SMALL_REGION_SIZE);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed";
    }
    
    // Manually modify used and mat_count to simulate usage
    region.used = 100;
    region.mat_count = 5;
    
    // Reset the region
    status = regreset(&region);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "regreset failed to reset region";
    }
    
    // Check if used space and matrix count were reset to zero
    if (region.used != 0 || region.mat_count != 0) {
        free(memory);
        return "regreset did not reset used space and matrix count to zero";
    }
    
    // Test with NULL parameter
    if (regreset(NULL) != MATRIX_NULL_POINTER) {
        free(memory);
        return "regreset failed to detect NULL region pointer";
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix allocation
const char* test_matalloc() {
    unsigned char* memory = (unsigned char*)malloc(MEDIUM_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, MEDIUM_REGION_SIZE);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed";
    }
    
    // Test allocation of a single matrix
    mat_t matrix;
    status = matalloc(&region, 10, 10, &matrix);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "matalloc failed to allocate a simple matrix";
    }
    
    // Check matrix properties
    if (matrix.row != 10 || matrix.col != 10) {
        free(memory);
        return "matalloc did not set matrix dimensions correctly";
    }
    
    // Check alignment of the matrix data
    if (!is_aligned(matrix.data)) {
        free(memory);
        return "matalloc did not align matrix data";
    }
    
    // Check stride handling - should be at least equal to cols and aligned
    if (matrix.stride < matrix.col) {
        free(memory);
        return "matalloc set stride smaller than column count";
    }
    
    if (matrix.stride % (SIMD_ALIGN / sizeof(float)) != 0) {
        free(memory);
        return "matalloc did not align stride properly for SIMD operations";
    }
    
    // Check that matrix count was incremented
    if (region.mat_count != 1) {
        free(memory);
        return "matalloc did not increment matrix count";
    }
    
    // Check that used space was updated
    size_t expected_size = matrix.row * matrix.stride * sizeof(float);
    // Allow for some padding due to alignment
    if (region.used < expected_size) {
        free(memory);
        return "matalloc did not update used space correctly";
    }
    
    // Test with invalid parameters
    if (matalloc(NULL, 10, 10, &matrix) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matalloc failed to detect NULL region pointer";
    }
    
    if (matalloc(&region, 10, 10, NULL) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matalloc failed to detect NULL matrix pointer";
    }
    
    if (matalloc(&region, 0, 10, &matrix) != MATRIX_DIMENSION_MISMATCH) {
        free(memory);
        return "matalloc failed to detect zero rows";
    }
    
    if (matalloc(&region, 10, 0, &matrix) != MATRIX_DIMENSION_MISMATCH) {
        free(memory);
        return "matalloc failed to detect zero columns";
    }
    
    // Test allocation until region is full
    size_t initial_used = region.used;
    size_t available = region.size - initial_used;
    size_t matrix_size = 100 * 100 * sizeof(float); // A large matrix
    
    // Calculate how many full matrices should fit in the remaining space
    size_t expected_matrices = available / matrix_size;
    if (expected_matrices == 0) {
        free(memory);
        return "Test error: region too small for allocation test";
    }
    
    size_t allocated_matrices = 0;
    while (1) {
        mat_t test_matrix;
        status = matalloc(&region, 100, 100, &test_matrix);
        if (status != MATRIX_SUCCESS) {
            break;
        }
        allocated_matrices++;
        
        // Safety check to prevent infinite loop
        if (allocated_matrices > expected_matrices + 10) {
            free(memory);
            return "matalloc did not detect region full (possible infinite loop)";
        }
    }
    
    // Should eventually return MATRIX_REGION_FULL
    if (status != MATRIX_REGION_FULL) {
        free(memory);
        return "matalloc did not return MATRIX_REGION_FULL when region is full";
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix creation with data
const char* test_matcreate() {
    unsigned char* memory = (unsigned char*)malloc(MEDIUM_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, MEDIUM_REGION_SIZE);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed";
    }
    
    // Create test data
    size_t rows = 5, cols = 7;
    float test_data[5 * 7];
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            test_data[i * cols + j] = (float)(i * 10 + j);
        }
    }
    
    // Test creation with data
    mat_t matrix;
    status = matcreate(&region, rows, cols, test_data, &matrix);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "matcreate failed to create matrix with data";
    }
    
    // Check matrix properties
    if (matrix.row != rows || matrix.col != cols) {
        free(memory);
        return "matcreate did not set matrix dimensions correctly";
    }
    
    // Check that data was copied correctly, accounting for stride
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float expected = test_data[i * cols + j];
            float actual = matrix.data[i * matrix.stride + j];
            if (expected != actual) {
                free(memory);
                return "matcreate did not copy data correctly";
            }
        }
    }
    
    // Test creation without data (NULL pointer)
    mat_t empty_matrix;
    status = matcreate(&region, rows, cols, NULL, &empty_matrix);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "matcreate failed to create matrix without data";
    }
    
    // Test with invalid parameters
    if (matcreate(NULL, rows, cols, test_data, &matrix) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matcreate failed to detect NULL region pointer";
    }
    
    if (matcreate(&region, 0, cols, test_data, &matrix) != MATRIX_DIMENSION_MISMATCH) {
        free(memory);
        return "matcreate failed to detect zero rows";
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix result allocation
const char* test_matresalloc() {
    unsigned char* memory = (unsigned char*)malloc(MEDIUM_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, MEDIUM_REGION_SIZE);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed";
    }
    
    // Allocate input matrices A and B for multiplication
    mat_t A, B, C;
    status = matalloc(&region, 10, 5, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: could not allocate matrix A";
    }
    
    status = matalloc(&region, 5, 8, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: could not allocate matrix B";
    }
    
    // Test result allocation
    status = matresalloc(&region, &A, &B, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "matresalloc failed to allocate result matrix";
    }
    
    // Check result matrix dimensions
    if (C.row != A.row || C.col != B.col) {
        free(memory);
        return "matresalloc did not set result matrix dimensions correctly";
    }
    
    // Test with dimension mismatch
    mat_t D;
    status = matalloc(&region, 7, 6, &D); // Dimensions don't match for multiplication
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: could not allocate matrix D";
    }
    
    status = matresalloc(&region, &A, &D, &C);
    if (status != MATRIX_DIMENSION_MISMATCH) {
        free(memory);
        return "matresalloc failed to detect dimension mismatch";
    }
    
    // Test with NULL parameters
    if (matresalloc(NULL, &A, &B, &C) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matresalloc failed to detect NULL region pointer";
    }
    
    if (matresalloc(&region, NULL, &B, &C) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matresalloc failed to detect NULL matrix A pointer";
    }
    
    if (matresalloc(&region, &A, NULL, &C) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matresalloc failed to detect NULL matrix B pointer";
    }
    
    if (matresalloc(&region, &A, &B, NULL) != MATRIX_NULL_POINTER) {
        free(memory);
        return "matresalloc failed to detect NULL matrix C pointer";
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test error string function
const char* test_strmaterr() {
    const char* success_str = strmaterr(MATRIX_SUCCESS);
    if (strcmp(success_str, "Success") != 0) {
        return "strmaterr did not return correct string for MATRIX_SUCCESS";
    }
    
    const char* null_ptr_str = strmaterr(MATRIX_NULL_POINTER);
    if (strcmp(null_ptr_str, "Null pointer provided") != 0) {
        return "strmaterr did not return correct string for MATRIX_NULL_POINTER";
    }
    
    const char* dim_mismatch_str = strmaterr(MATRIX_DIMENSION_MISMATCH);
    if (strcmp(dim_mismatch_str, "Matrix dimension mismatch") != 0) {
        return "strmaterr did not return correct string for MATRIX_DIMENSION_MISMATCH";
    }
    
    const char* region_full_str = strmaterr(MATRIX_REGION_FULL);
    if (strcmp(region_full_str, "Memory region is full") != 0) {
        return "strmaterr did not return correct string for MATRIX_REGION_FULL";
    }
    
    const char* invalid_region_str = strmaterr(MATRIX_INVALID_REGION);
    if (strcmp(invalid_region_str, "Invalid memory region") != 0) {
        return "strmaterr did not return correct string for MATRIX_INVALID_REGION";
    }
    
    // Test unknown error code
    const char* unknown_str = strmaterr(200); // An undefined error code
    if (strcmp(unknown_str, "Unknown matrix error") != 0) {
        return "strmaterr did not return correct string for unknown error";
    }
    
    return TEST_PASSED;
}

// Test for multiple allocations and tracking
const char* test_multiple_allocations() {
    unsigned char* memory = (unsigned char*)malloc(LARGE_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, LARGE_REGION_SIZE);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed";
    }
    
    // Allocate a series of matrices with different dimensions
    const size_t num_matrices = 10;
    mat_t matrices[num_matrices];
    
    for (size_t i = 0; i < num_matrices; i++) {
        size_t rows = (i + 1) * 5;
        size_t cols = (i + 1) * 3;
        
        status = matalloc(&region, rows, cols, &matrices[i]);
        if (status != MATRIX_SUCCESS) {
            free(memory);
            return "Failed to allocate matrix during multiple allocation test";
        }
        
        // Fill the matrix with a recognizable pattern
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                matrices[i].data[r * matrices[i].stride + c] = (float)(i * 1000 + r * 100 + c);
            }
        }
    }
    
    // Verify matrix count
    if (region.mat_count != num_matrices) {
        free(memory);
        return "Region did not correctly track the number of allocated matrices";
    }
    
    // Verify each matrix still contains the correct data
    for (size_t i = 0; i < num_matrices; i++) {
        size_t rows = (i + 1) * 5;
        size_t cols = (i + 1) * 3;
        
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                float expected = (float)(i * 1000 + r * 100 + c);
                float actual = matrices[i].data[r * matrices[i].stride + c];
                
                if (expected != actual) {
                    free(memory);
                    return "Matrix data was corrupted during multiple allocations";
                }
            }
        }
    }
    
    // Reset the region and reallocate to ensure cleanup works
    status = regreset(&region);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Failed to reset region after multiple allocations";
    }
    
    // Try a single large allocation after reset
    status = matalloc(&region, 100, 100, &matrices[0]);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Failed to allocate matrix after region reset";
    }
    
    // Verify matrix count was reset
    if (region.mat_count != 1) {
        free(memory);
        return "Region did not correctly reset the matrix count";
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test alignment stress test with odd-sized allocations
const char* test_alignment_stress() {
    unsigned char* memory = (unsigned char*)malloc(MEDIUM_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    // Intentionally create a misaligned base pointer
    unsigned char* misaligned_base = memory + 1;
    
    mat_region_t region;
    mat_status_t status = reginit(&region, misaligned_base, MEDIUM_REGION_SIZE - 1);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed with misaligned pointer";
    }
    
    // Allocate matrices with prime number dimensions to stress alignment
    const size_t dims[] = {3, 5, 7, 11, 13, 17, 19, 23};
    const size_t num_matrices = sizeof(dims) / sizeof(dims[0]);
    mat_t matrices[num_matrices];
    
    for (size_t i = 0; i < num_matrices; i++) {
        status = matalloc(&region, dims[i], dims[(i + 1) % num_matrices], &matrices[i]);
        if (status != MATRIX_SUCCESS) {
            free(memory);
            return "Failed to allocate matrix during alignment stress test";
        }
        
        // Verify alignment of data pointer
        if (!is_aligned(matrices[i].data)) {
            free(memory);
            return "Matrix data not aligned during alignment stress test";
        }
        
        // Verify stride alignment
        if (matrices[i].stride % (SIMD_ALIGN / sizeof(float)) != 0) {
            free(memory);
            return "Matrix stride not aligned during alignment stress test";
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

static void print_matrix(const mat_t *m) {
    if (!m) {
        printf("Matrix is NULL.\n");
        return;
    }

    // Print matrix metadata
    printf("Matrix: %zu x %zu (stride: %zu)\n", m->row, m->col, m->stride);
    printf("Data:\n");

    // Print matrix data
    for (size_t i = 0; i < m->row; i++) {
        for (size_t j = 0; j < m->col; j++) {
            // Access element using stride for proper alignment
            printf("%8.4f ", m->data[i * m->stride + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Test matrix data integrity
const char* test_matrix_data_integrity() {
    unsigned char* memory = (unsigned char*)malloc(MEDIUM_REGION_SIZE);
    if (!memory) return "Failed to allocate memory for test";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, MEDIUM_REGION_SIZE);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Setup failed: reginit failed";
    }
    
    // Test different matrix sizes to verify data integrity
    struct test_case {
        size_t rows;
        size_t cols;
        const char* name;
    };
    
    struct test_case test_cases[] = {
        {4, 4, "Small square matrix"},
        {10, 5, "Medium rectangular matrix (rows > cols)"},
        {3, 12, "Medium rectangular matrix (cols > rows)"},
        {1, 20, "Row vector"},
        {15, 1, "Column vector"},
        {50, 50, "Large square matrix"}
    };
    
    const int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int t = 0; t < num_test_cases; t++) {
        size_t rows = test_cases[t].rows;
        size_t cols = test_cases[t].cols;
        
        // Create test data with distinctive pattern
        float* test_data = (float*)malloc(rows * cols * sizeof(float));
        if (!test_data) {
            free(memory);
            return "Failed to allocate test data";
        }
        
        // Fill with a pattern where each element is unique and easily verifiable
        // We'll use: value = row*1000 + col*10 + (row+col)%10
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                test_data[i * cols + j] = (float)(i * 1000 + j * 10 + (i + j) % 10);
            }
        }
        
        // First test: Create matrix with data
        mat_t matrix;
        status = matcreate(&region, rows, cols, test_data, &matrix);
        if (status != MATRIX_SUCCESS) {
            free(test_data);
            free(memory);
            return "matcreate failed to create matrix with data";
        }
        
        // Verify matrix dimensions
        if (matrix.row != rows || matrix.col != cols) {
            free(test_data);
            free(memory);
            return "matcreate did not set matrix dimensions correctly";
        }
        
        // Verify each element of the matrix, accounting for stride
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                float expected = test_data[i * cols + j];
                float actual = matrix.data[i * matrix.stride + j];
                
                if (expected != actual) {
                    printf("Data mismatch in %s at [%zu,%zu]: expected %.1f, got %.1f\n", 
                           test_cases[t].name, i, j, expected, actual);
                    free(test_data);
                    free(memory);
                    return "matcreate did not copy data correctly";
                }
            }
        }
        
        // Second test: Verify data points to correct memory location in region
        if (matrix.data < region.memory || 
            (unsigned char*)matrix.data >= region.memory + region.size) {
            free(test_data);
            free(memory);
            return "Matrix data pointer is outside the allocated region";
        }
        
        // Check for alignment of data
        if (!is_aligned(matrix.data)) {
            free(test_data);
            free(memory);
            return "Matrix data is not properly aligned";
        }
        
        // Third test: Modify data and ensure it changes the correct elements
        // Change a few scattered elements in different patterns
        size_t test_indices[][2] = {
            {0, 0},                    // First element
            {rows - 1, cols - 1},      // Last element
            {rows / 2, cols / 2},      // Middle element
            {0, cols - 1},             // Top-right corner
            {rows - 1, 0}              // Bottom-left corner
        };
        const int num_test_indices = sizeof(test_indices) / sizeof(test_indices[0]);
        
        // Create a copy of the original data for comparison
        float* original_data_copy = (float*)malloc(rows * cols * sizeof(float));
        if (!original_data_copy) {
            free(test_data);
            free(memory);
            return "Failed to allocate memory for original data copy";
        }
        memcpy(original_data_copy, test_data, rows * cols * sizeof(float));

        // Track which elements we've modified
        bool* modified = (bool*)calloc(rows * cols, sizeof(bool));
        if (!modified) {
            free(original_data_copy);
            free(test_data);
            free(memory);
            return "Failed to allocate memory for modification tracking";
        }

        // Only test valid indices
        for (int idx = 0; idx < num_test_indices; idx++) {
            size_t r = test_indices[idx][0];
            size_t c = test_indices[idx][1];
            
            if (r < rows && c < cols) {
                //print_matrix(&matrix);
                float new_value = -999.0f - (float)idx;
                matrix.data[r * matrix.stride + c] = new_value;
                //print_matrix(&matrix);
                
                // Mark this element as modified
                modified[r * cols + c] = true;
                
                // Verify the change took effect
                if (matrix.data[r * matrix.stride + c] != new_value) {
                    free(modified);
                    free(original_data_copy);
                    free(test_data);
                    free(memory);
                    return "Failed to modify matrix element";
                }
                
                // Verify that only the intended elements were changed
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        // Skip elements we've intentionally modified
                        if (modified[i * cols + j]) continue;
                        
                        float expected = original_data_copy[i * cols + j];
                        float actual = matrix.data[i * matrix.stride + j];
                        
                        if (expected != actual) {
                            printf("Unintended data change at [%zu,%zu] after modifying [%zu,%zu]\n",
                                i, j, r, c);
                            free(modified);
                            free(original_data_copy);
                            free(test_data);
                            free(memory);
                            return "Modifying one element affected other elements";
                        }
                    }
                }
            }
        }

        free(modified);
        free(original_data_copy);
        
        free(test_data);
    }
    
    // Test edge cases: very small and very large values
    float edge_data[] = {
        -1e30f, -1e20f, -1e10f, -1.0f, -1e-10f, -1e-20f, -1e-30f,
        0.0f,
        1e-30f, 1e-20f, 1e-10f, 1.0f, 1e10f, 1e20f, 1e30f,
        INFINITY, -INFINITY, NAN
    };
    size_t edge_size = sizeof(edge_data) / sizeof(edge_data[0]);
    
    mat_t edge_matrix;
    status = matcreate(&region, 1, edge_size, edge_data, &edge_matrix);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "matcreate failed to create matrix with edge case data";
    }
    
    // Verify edge case values
    for (size_t j = 0; j < edge_size; j++) {
        float expected = edge_data[j];
        float actual = edge_matrix.data[j];
        
        // Special handling for NaN (NaN != NaN)
        if (isnan(expected) && isnan(actual)) {
            continue; // Both are NaN, consider it a match
        }
        
        if (expected != actual) {
            printf("Edge case data mismatch at [0,%zu]: expected %.10g, got %.10g\n", 
                   j, expected, actual);
            free(memory);
            return "matcreate did not copy edge case data correctly";
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

int main() {
    // Define all tests
    struct {
        const char* name;
        const char* (*test_func)();
    } tests[] = {
        {"Region Initialization", test_reginit},
        {"Region Reset", test_regreset},
        {"Matrix Allocation", test_matalloc},
        {"Matrix Creation", test_matcreate},
        {"Matrix Result Allocation", test_matresalloc},
        {"Error String", test_strmaterr},
        {"Multiple Allocations", test_multiple_allocations},
        {"Alignment Stress Test", test_alignment_stress},
        {"Matrix Data Integrity Test", test_matrix_data_integrity}
    };
    
    int test_count = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    
    printf("Running %d tests for matrix memory pool...\n\n", test_count);
    
    for (int i = 0; i < test_count; i++) {
        printf("Test %d: %s... ", i + 1, tests[i].name);
        const char* result = tests[i].test_func();
        printf("%s\n", result);
        
        if (strcmp(result, TEST_PASSED) == 0) {
            passed++;
        } else {
            printf("  Error: %s\n", result);
        }
    }
    
    printf("\nTest Results: %d/%d tests passed\n", passed, test_count);
    
    return (passed == test_count) ? 0 : 1;
}