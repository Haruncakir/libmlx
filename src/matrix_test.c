// (AVX) gcc -O3 -fopenmp -mavx -Wall matrix_test.c -o simd_matrix_example
// (NEON) gcc -O3 -fopenmp -mfpu=neon -Wall matrix_test.c -o simd_matrix_example
// -march=native: Optimizes for your specific CPU (e.g., uses AVX2 if available)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

// Example demonstrating zero-copy operations with caller-provided buffers
void zero_copy_example() {
    printf("\n=== Zero-Copy Matrix Operations Example ===\n");
    
    // Dimensions
    size_t rows = 4;
    size_t cols = 8;
    
    // Calculate required stride for alignment
    size_t stride = cols;
    if (cols % VECTOR_SIZE != 0) {
        stride = (cols / VECTOR_SIZE + 1) * VECTOR_SIZE;
    }
    
    // Allocate aligned buffers that the caller manages
    ALIGNED float buffer_a[4 * 8] = {0};
    ALIGNED float buffer_b[4 * 8] = {0};
    ALIGNED float buffer_c[4 * 8] = {0};
    
    // Initialize matrices with caller-provided buffers
    Matrix a = matrix_create_with_buffer(buffer_a, rows, cols, stride);
    Matrix b = matrix_create_with_buffer(buffer_b, rows, cols, stride);
    Matrix c = matrix_create_with_buffer(buffer_c, rows, cols, stride);
    
    // Fill matrices with test data
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix_set(&a, i, j, (float)(i + j));
            matrix_set(&b, i, j, (float)(i * j + 1));
        }
    }
    
    // Perform matrix addition using SIMD
    matrix_add(&c, &a, &b);
    
    // Print results
    printf("Matrix A:\n");
    matrix_print(&a);
    
    printf("\nMatrix B:\n");
    matrix_print(&b);
    
    printf("\nMatrix C = A + B:\n");
    matrix_print(&c);
    
    // No need to call matrix_destroy() since we're using caller-provided buffers
}

// Example demonstrating dot product and matrix-vector multiplication
void vector_operations_example() {
    printf("\n=== Vector Operations Example ===\n");
    
    // Create test data
    size_t size = 8;
    ALIGNED float vec_a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    ALIGNED float vec_b[8] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    
    // Calculate dot product
    float dot = vector_dot_product(vec_a, vec_b, size);
    
    printf("Vector A: ");
    for (size_t i = 0; i < size; i++) {
        printf("%.1f ", vec_a[i]);
    }
    printf("\n");
    
    printf("Vector B: ");
    for (size_t i = 0; i < size; i++) {
        printf("%.1f ", vec_b[i]);
    }
    printf("\n");
    
    printf("Dot product A·B: %.1f\n", dot);
    
    // Matrix-vector multiplication example
    printf("\n=== Matrix-Vector Multiplication Example ===\n");
    
    // Create a 4x4 matrix
    Matrix m = matrix_create(4, 4);
    ALIGNED float vec_x[4] = {1.0, 2.0, 3.0, 4.0};
    ALIGNED float vec_y[4] = {0.0, 0.0, 0.0, 0.0};
    
    // Fill matrix with values
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            matrix_set(&m, i, j, (float)(i + j + 1));
        }
    }
    
    printf("Matrix A:\n");
    matrix_print(&m);
    
    printf("\nVector x: ");
    for (size_t i = 0; i < 4; i++) {
        printf("%.1f ", vec_x[i]);
    }
    printf("\n");
    
    // Perform matrix-vector multiplication
    matrix_vector_multiply(vec_y, &m, vec_x);
    
    printf("Result y = A * x: ");
    for (size_t i = 0; i < 4; i++) {
        printf("%.1f ", vec_y[i]);
    }
    printf("\n");
    
    // Clean up
    matrix_destroy(&m);
}

// Example showing how to create and use matrix views
void matrix_view_example() {
    printf("\n=== Matrix View Example ===\n");
    
    // Create a 6x6 matrix
    Matrix m = matrix_create(6, 6);
    
    // Fill with consecutive numbers
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            matrix_set(&m, i, j, (float)(i * m.cols + j));
        }
    }
    
    printf("Original matrix:\n");
    matrix_print(&m);
    
    // Create a 3x3 view starting at position (1,1)
    MatrixView view = matrix_view_create(&m, 1, 1, 3, 3);
    
    printf("\nMatrix view (3x3 from position 1,1):\n");
    for (size_t i = 0; i < view.num_rows; i++) {
        for (size_t j = 0; j < view.num_cols; j++) {
            printf("%8.4f ", matrix_view_get(&view, i, j));
        }
        printf("\n");
    }
    
    // Modify view and see how it affects the original matrix
    printf("\nSetting view values to 99...\n");
    for (size_t i = 0; i < view.num_rows; i++) {
        for (size_t j = 0; j < view.num_cols; j++) {
            matrix_view_set(&view, i, j, 99.0f);
        }
    }
    
    printf("\nOriginal matrix after view modification:\n");
    matrix_print(&m);
    
    // Clean up
    matrix_destroy(&m);
}

// Benchmark for measuring performance
void performance_benchmark() {
    printf("\n=== Performance Benchmark ===\n");
    
    // Matrix dimensions for benchmark
    size_t size = 256;
    
    // Create matrices
    Matrix a = matrix_create(size, size);
    Matrix b = matrix_create(size, size);
    Matrix c = matrix_create(size, size);
    
    // Fill with random data
    srand(42);  // Use fixed seed for reproducibility
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            matrix_set(&a, i, j, (float)rand() / RAND_MAX);
            matrix_set(&b, i, j, (float)rand() / RAND_MAX);
        }
    }
    
    // Benchmark matrix addition
    clock_t start = clock();
    matrix_add(&c, &a, &b);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Matrix addition (%zux%zu): %.6f seconds\n", size, size, time_taken);
    
    // Clean up
    matrix_destroy(&a);
    matrix_destroy(&b);
    matrix_destroy(&c);
}

int main() {
    printf("SIMD Matrix Library Example\n");
    printf("SIMD Vector Size: %d floats\n", VECTOR_SIZE);
    
    // Run examples
    zero_copy_example();
    vector_operations_example();
    matrix_view_example();
    performance_benchmark();
    
    return 0;
}