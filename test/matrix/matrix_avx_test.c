#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../../include/matrix/matpool.h"
#include "../../include/matrix/matrix_config.h"
#include "../../include/matrix/matrix.h"

#define TEST_PASSED "\033[32mPASSED\033[0m"
#define TEST_FAILED "\033[31mFAILED\033[0m"

#define EPSILON 1e-5f
#define MAX_TEST_SIZE 100

// Helper function to check if two float values are close enough
int float_equals(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

// Helper function to print a matrix for debugging
void print_matrix(const char* name, const mat_t* mat) {
    printf("Matrix %s (%zu x %zu):\n", name, mat->row, mat->col);
    for (size_t i = 0; i < mat->row; i++) {
        for (size_t j = 0; j < mat->col; j++) {
            printf("%8.3f ", mat->data[i * mat->stride + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Test matrix addition
const char* test_matadd() {
    // Allocate memory for the region
    size_t region_size = 3 * MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + 3 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B, C;
    size_t rows = 10, cols = 10;
    
    // Initialize matrices with test data
    status = matalloc(&region, rows, cols, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, rows, cols, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    status = matalloc(&region, rows, cols, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix C allocation failed";
    }
    
    // Fill matrices A and B with test data
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A.data[i * A.stride + j] = (float)(i + j);
            B.data[i * B.stride + j] = (float)(i - j);
        }
    }
    
    // Test addition
    status = matadd(&A, &B, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix addition failed";
    }
    
    // Verify the result
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float expected = (float)(i + j) + (float)(i - j);
            float actual = C.data[i * C.stride + j];
            if (!float_equals(actual, expected)) {
                printf("Mismatch at (%zu, %zu): Expected %.3f, got %.3f\n", i, j, expected, actual);
                free(memory);
                return "Matrix addition result verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix subtraction
const char* test_matsub() {
    // Allocate memory for the region
    size_t region_size = 3 * MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + 3 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B, C;
    size_t rows = 10, cols = 10;
    
    // Initialize matrices with test data
    status = matalloc(&region, rows, cols, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, rows, cols, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    status = matalloc(&region, rows, cols, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix C allocation failed";
    }
    
    // Fill matrices A and B with test data
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A.data[i * A.stride + j] = (float)(i + j);
            B.data[i * B.stride + j] = (float)(j);
        }
    }
    
    // Test subtraction
    status = matsub(&A, &B, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix subtraction failed";
    }
    
    // Verify the result
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float expected = (float)(i + j) - (float)(j);
            float actual = C.data[i * C.stride + j];
            if (!float_equals(actual, expected)) {
                printf("Mismatch at (%zu, %zu): Expected %.3f, got %.3f\n", i, j, expected, actual);
                free(memory);
                return "Matrix subtraction result verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix multiplication
const char* test_matmul() {
    // Allocate memory for the region
    size_t region_size = 3 * MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + 3 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B, C;
    size_t rows_a = 5, cols_a = 4, cols_b = 3;
    
    // Initialize matrices with test data
    status = matalloc(&region, rows_a, cols_a, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, cols_a, cols_b, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    status = matalloc(&region, rows_a, cols_b, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix C allocation failed";
    }
    
    // Fill matrices A and B with test data
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            A.data[i * A.stride + j] = (float)(i + j + 1);
        }
    }
    
    for (size_t i = 0; i < cols_a; i++) {
        for (size_t j = 0; j < cols_b; j++) {
            B.data[i * B.stride + j] = (float)(i * j + 1);
        }
    }
    
    // Test multiplication
    status = matmul(&A, &B, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix multiplication failed";
    }
    
    // Verify the result
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_b; j++) {
            float expected = 0.0f;
            for (size_t k = 0; k < cols_a; k++) {
                expected += A.data[i * A.stride + k] * B.data[k * B.stride + j];
            }
            float actual = C.data[i * C.stride + j];
            if (!float_equals(actual, expected)) {
                printf("Matmul mismatch at (%zu, %zu): Expected %.3f, got %.3f\n", i, j, expected, actual);
                free(memory);
                return "Matrix multiplication result verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix transpose
const char* test_mattranspose() {
    // Allocate memory for the region
    size_t region_size = 2 * MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + 2 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B;
    size_t rows_a = 10, cols_a = 5;
    
    // Initialize matrices with test data
    status = matalloc(&region, rows_a, cols_a, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, cols_a, rows_a, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    // Fill matrix A with test data
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            A.data[i * A.stride + j] = (float)(i * cols_a + j);
        }
    }
    
    // Test transpose
    status = mattranspose(&A, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix transpose failed";
    }
    
    // Verify the result
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            float expected = A.data[i * A.stride + j];
            float actual = B.data[j * B.stride + i];
            if (!float_equals(actual, expected)) {
                printf("Transpose mismatch at (%zu, %zu): Expected %.3f, got %.3f\n", j, i, expected, actual);
                free(memory);
                return "Matrix transpose result verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test dot product
const char* test_matdot() {
    size_t length = 100;
    float* a = (float*)malloc(length * sizeof(float));
    float* b = (float*)malloc(length * sizeof(float));
    
    if (!a || !b) {
        if (a) free(a);
        if (b) free(b);
        return "Memory allocation failed";
    }
    
    // Fill vectors with test data
    for (size_t i = 0; i < length; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)(length - i);
    }
    
    // Calculate expected result
    float expected = 0.0f;
    for (size_t i = 0; i < length; i++) {
        expected += a[i] * b[i];
    }
    
    // Test dot product
    float actual = matdot(a, b, length);
    
    free(a);
    free(b);
    
    // Verify the result
    if (!float_equals(actual, expected)) {
        printf("Dot product mismatch: Expected %.3f, got %.3f\n", expected, actual);
        return "Dot product result verification failed";
    }
    
    return TEST_PASSED;
}

// Test matrix-vector multiplication
const char* test_matvecmul() {
    // Allocate memory for the region
    size_t region_size = MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrix
    mat_t A;
    size_t rows = 10, cols = 5;
    
    // Initialize matrix with test data
    status = matalloc(&region, rows, cols, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    // Create test vectors
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));
    
    if (!x || !y) {
        free(memory);
        if (x) free(x);
        if (y) free(y);
        return "Vector allocation failed";
    }
    
    // Fill matrix and vector with test data
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            A.data[i * A.stride + j] = (float)(i + j + 1);
        }
    }
    
    for (size_t j = 0; j < cols; j++) {
        x[j] = (float)(j + 1);
    }
    
    // Test matrix-vector multiplication
    status = matvecmul(&A, x, y);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        free(x);
        free(y);
        return "Matrix-vector multiplication failed";
    }
    
    // Verify the result
    for (size_t i = 0; i < rows; i++) {
        float expected = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            expected += A.data[i * A.stride + j] * x[j];
        }
        if (!float_equals(y[i], expected)) {
            printf("Matvecmul mismatch at index %zu: Expected %.3f, got %.3f\n", i, expected, y[i]);
            free(memory);
            free(x);
            free(y);
            return "Matrix-vector multiplication result verification failed";
        }
    }
    
    free(memory);
    free(x);
    free(y);
    return TEST_PASSED;
}

// Test matrix reshape
const char* test_matreshape() {
    // Allocate memory for the region
    size_t region_size = 2 * MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + 2 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B;
    size_t rows_a = 4, cols_a = 6;
    size_t rows_b = 6, cols_b = 4; // Transposed dimensions
    
    // Initialize matrices with test data
    status = matalloc(&region, rows_a, cols_a, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, rows_b, cols_b, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    // Fill matrix A with sequential values
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            A.data[i * A.stride + j] = (float)(i * cols_a + j);
        }
    }
    
    // Test reshape
    status = matreshape(&A, rows_b, cols_b, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix reshape failed";
    }
    
    // Verify the result - reshape should preserve the element order when flattened
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            size_t linear_idx = i * cols_a + j;
            size_t new_i = linear_idx / cols_b;
            size_t new_j = linear_idx % cols_b;
            
            float expected = A.data[i * A.stride + j];
            float actual = B.data[new_i * B.stride + new_j];
            
            if (!float_equals(actual, expected)) {
                printf("Reshape mismatch at (%zu, %zu)->(%zu, %zu): Expected %.3f, got %.3f\n", 
                       i, j, new_i, new_j, expected, actual);
                free(memory);
                return "Matrix reshape result verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test matrix block reshape
const char* test_matreshapeblock() {
    // Allocate memory for the region
    size_t region_size = 2 * MAX_TEST_SIZE * MAX_TEST_SIZE * sizeof(float) + 2 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B;
    size_t rows_a = 8, cols_a = 6;
    size_t rows_b = 12, cols_b = 4; // Different shape with same total elements
    
    // Initialize matrices with test data
    status = matalloc(&region, rows_a, cols_a, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, rows_b, cols_b, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    // Fill matrix A with sequential values
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            A.data[i * A.stride + j] = (float)(i * cols_a + j);
        }
    }
    
    // Test block reshape
    status = matreshapeblock(&A, rows_b, cols_b, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix block reshape failed";
    }
    
    // Verify the result - reshape should preserve the element order when flattened
    for (size_t i = 0; i < rows_a; i++) {
        for (size_t j = 0; j < cols_a; j++) {
            size_t linear_idx = i * cols_a + j;
            size_t new_i = linear_idx / cols_b;
            size_t new_j = linear_idx % cols_b;
            
            float expected = A.data[i * A.stride + j];
            float actual = B.data[new_i * B.stride + new_j];
            
            if (!float_equals(actual, expected)) {
                printf("Block reshape mismatch at (%zu, %zu)->(%zu, %zu): Expected %.3f, got %.3f\n", 
                       i, j, new_i, new_j, expected, actual);
                free(memory);
                return "Matrix block reshape result verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

// Test all functions with large matrices to verify AVX optimization
const char* test_large_matrices() {
    // Allocate memory for the region
    size_t large_size = 64; // Large enough to trigger AVX code paths
    size_t region_size = 3 * large_size * large_size * sizeof(float) + 3 * sizeof(mat_t) + SIMD_ALIGN;
    void* memory = malloc(region_size);
    if (!memory) return "Memory allocation failed";
    
    mat_region_t region;
    mat_status_t status = reginit(&region, memory, region_size);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Region initialization failed";
    }
    
    // Create test matrices
    mat_t A, B, C;
    
    // Initialize matrices with test data
    status = matalloc(&region, large_size, large_size, &A);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix A allocation failed";
    }
    
    status = matalloc(&region, large_size, large_size, &B);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix B allocation failed";
    }
    
    status = matalloc(&region, large_size, large_size, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Matrix C allocation failed";
    }
    
    // Fill matrices with test data
    for (size_t i = 0; i < large_size; i++) {
        for (size_t j = 0; j < large_size; j++) {
            A.data[i * A.stride + j] = (float)(i + j);
            B.data[i * B.stride + j] = (float)(i - j);
        }
    }
    
    // Test addition with large matrices
    status = matadd(&A, &B, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Large matrix addition failed";
    }
    
    // Spot check some values
    for (size_t i = 0; i < large_size; i += large_size/8) {
        for (size_t j = 0; j < large_size; j += large_size/8) {
            float expected = (float)(i + j) + (float)(i - j);
            float actual = C.data[i * C.stride + j];
            if (!float_equals(actual, expected)) {
                printf("Large matrix add mismatch at (%zu, %zu): Expected %.3f, got %.3f\n", 
                       i, j, expected, actual);
                free(memory);
                return "Large matrix addition verification failed";
            }
        }
    }
    
    // Test subtraction with large matrices
    status = matsub(&A, &B, &C);
    if (status != MATRIX_SUCCESS) {
        free(memory);
        return "Large matrix subtraction failed";
    }
    
    // Spot check some values
    for (size_t i = 0; i < large_size; i += large_size/8) {
        for (size_t j = 0; j < large_size; j += large_size/8) {
            float expected = (float)(i + j) - (float)(i - j);
            float actual = C.data[i * C.stride + j];
            if (!float_equals(actual, expected)) {
                printf("Large matrix sub mismatch at (%zu, %zu): Expected %.3f, got %.3f\n", 
                       i, j, expected, actual);
                free(memory);
                return "Large matrix subtraction verification failed";
            }
        }
    }
    
    free(memory);
    return TEST_PASSED;
}

int main() {
    // Register all tests
    struct {
        const char* name;
        const char* (*test_func)();
    } tests[] = {
        {"Matrix Addition", test_matadd},
        {"Matrix Subtraction", test_matsub},
        {"Matrix Multiplication", test_matmul},
        {"Matrix Transpose", test_mattranspose},
        {"Dot Product", test_matdot},
        {"Matrix-Vector Multiplication", test_matvecmul},
        {"Matrix Reshape", test_matreshape},
        {"Matrix Block Reshape", test_matreshapeblock},
        {"Large Matrix Operations", test_large_matrices},
    };
    
    int test_count = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    
    printf("Running %d tests...\n\n", test_count);
    
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