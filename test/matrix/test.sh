#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ../exe

# Compile source files
gcc -g -mavx -Wall -Wextra -c ../../src/common/matrix_common.c -o ../exe/matrix_common.o || { echo "Compilation failed for matrix_common.c"; exit 1; }
gcc -g -mavx -Wall -Wextra -c ../../src/common/matpool.c -o ../exe/matpool.o || { echo "Compilation failed for matpool.c"; exit 1; }
gcc -g -mavx -Wall -Wextra -c ../../src/arch/avx/matrix_avx.c -o ../exe/matrix_avx.o || { echo "Compilation failed for matrix_avx.c"; exit 1; }

# Compile test file (for matrix_avx test for now)
gcc -g -mavx -Wall -Wextra -c matrix_avx_test.c -o ../exe/matrix_avx_test.o || { echo "Compilation failed for matrix_avx_test.c"; exit 1; }

# Link object files to create the executable
gcc -o ../exe/matrix_avx_test ../exe/matrix_common.o ../exe/matpool.o ../exe/matrix_avx.o ../exe/matrix_avx_test.o || { echo "Linking failed"; exit 1; }

# Run the executable
./../exe/matrix_avx_test