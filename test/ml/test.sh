#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ../exe

# Compile source files
gcc -g -mavx -Wall -Wextra -c ../../src/common/matrix_common.c -o ../exe/matrix_common.o || { echo "Compilation failed for matrix_common.c"; exit 1; }
gcc -g -mavx -Wall -Wextra -c ../../src/common/matpool.c -o ../exe/matpool.o || { echo "Compilation failed for matpool.c"; exit 1; }
gcc -g -mavx -Wall -Wextra -c ../../src/arch/avx/matrix_avx.c -o ../exe/matrix_avx.o || { echo "Compilation failed for matrix_avx.c"; exit 1; }
gcc -g -mavx -Wall -Wextra -c ../../src/ml/logistic.c -o ../exe/logreg.o || { echo "Compilation failed for logistic.c"; exit 1; }

# Compile test file (for logistic regression test for now)
gcc -g -mavx -Wall -Wextra -c logreg_test.c -o ../exe/logreg_test.o || { echo "Compilation failed for logreg_test.c"; exit 1; }

# Link object files to create the executable
gcc -o ../exe/logreg_test ../exe/matrix_common.o ../exe/matpool.o ../exe/matrix_avx.o ../exe/logreg.o ../exe/logreg_test.o || { echo "Linking failed"; exit 1; }

gcc -g -mavx -Wall -Wextra -c activations_test.c -o ../exe/activations_test.o || { echo "Compilation failed for activations_test.c"; exit 1; }
gcc -o ../exe/activations_test ../exe/activations_test.o -lm

# Run the executable
./../exe/logreg_test
# ./../exe/activations_test