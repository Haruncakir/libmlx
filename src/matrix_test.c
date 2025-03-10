// (AVX) gcc -O3 -fopenmp -mavx -Wall matrix_test.c -o simd_matrix_example
// (NEON) gcc -O3 -fopenmp -mfpu=neon -Wall matrix_test.c -o simd_matrix_example
// -march=native: Optimizes for your specific CPU (e.g., uses AVX2 if available)

