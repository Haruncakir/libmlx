# libmlx
---
## Embedded Machine Learning Library

## Embedded SIMD Matrix Library

A high-performance matrix library optimized for embedded systems with support for x86 AVX, ARM NEON, and fallback scalar implementations.

### Features

- **Zero-copy execution** using caller-provided memory buffers
- **SIMD optimizations** for both AVX and NEON instruction sets
- **Memory-efficient** with proper alignment for SIMD operations
- **Modular architecture** with separate implementations per platform
- **Flexible compilation** for different target architectures
- **Memory pool** for efficient region-based memory management

### Architecture

The library is designed with a modular architecture to support different instruction sets while maintaining a small code footprint:

- `matrix.h` - Common API definitions
- `matrix_config.h` - Platform detection and configuration
- `matrix_common.c` - Common utility functions for all platforms
- `matrix_avx.c` - AVX-specific optimized implementations
- `matrix_neon.c` - NEON-specific optimized implementations
- `matrix_scalar.c` - Fallback scalar implementations
- `matpool.h` - Memory pool management API
- `matpool.c` - Memory pool implementation

### Building the Library

#### Quick Start

```bash
# Auto-detect architecture and build
make

# Build for specific architectures
make avx     # Build for Intel AVX
make avx2    # Build for Intel AVX2
make neon    # Build for ARM NEON
make scalar  # Force scalar implementation

# Clean build artifacts
make clean
```

### Advanced Configuration

```bash
# Set custom compiler
make CC=clang

# Add custom compiler flags
make CFLAGS="-Wall -O3 -ffast-math"
```

## Using the Library

### Basic Matrix Operations

```c
#include "matrix.h"
#include "matpool.h"

// Create memory region for matrices
unsigned char memory[1024 * 1024];
mat_region_t region;
reginit(&region, memory, sizeof(memory));

// Allocate matrices
mat_t a, b, c;
matalloc(&region, 100, 100, &a);
matalloc(&region, 100, 100, &b);
matalloc(&region, 100, 100, &c);

// Fill with values
mat_fill(&a, 1.0f);
mat_fill(&b, 2.0f);

// Perform operations
mat_multiply(&a, &b, &c);
```

### Zero-Copy Matrix Operations

```c
// Use pre-allocated memory
float buffer_a[16] = {1.0f, 2.0f, 3.0f, 4.0f, /* ... */};
float buffer_b[16] = {5.0f, 6.0f, 7.0f, 8.0f, /* ... */};
float buffer_c[16] = {0};

// Create matrices with caller-provided buffers
mat_t a, b, c;
a.data = buffer_a;
a.row = 4;
a.col = 4;
a.stride = 4;

b.data = buffer_b;
b.row = 4;
b.col = 4;
b.stride = 4;

c.data = buffer_c;
c.row = 4;
c.col = 4;
c.stride = 4;

// Perform operation
mat_add(&a, &b, &c);
```

### Memory Alignment

The library ensures proper memory alignment for SIMD operations:

- AVX: 32-byte alignment (256-bit registers)
- NEON: 16-byte alignment (128-bit registers)
- Scalar: 4-byte alignment (regular float operations)

This is handled automatically by the memory pool implementation.

### Performance Considerations

- Use the memory pool whenever possible for optimal alignment
- For maximum performance, compile with the specific architecture target
- For large matrices, OpenMP parallelization can be enabled in your application
- Optimal matrix dimensions are multiples of:
  - 8 floats for AVX (256 bits / 32 bytes)
  - 4 floats for NEON (128 bits / 16 bytes)

### License

[Apache License](LICENSE)

---

4️⃣ First Steps in Development

💡 Step 1: Implement Fast Matrix Operations

    Why? Almost all ML models rely on matrix multiplications, so we optimize those first.
    How? Use SIMD (AVX, NEON) for fast dot products, matrix-vector multiplication, etc.

💡 Step 2: Implement Simple Model Inference

    Start with logistic regression & decision trees (easy & lightweight).
    Example: y = Wx + b (Matrix multiplication → Apply Activation Function).

💡 Step 3: Build Model Loading System

    Read tiny models in a binary format (so no huge frameworks like TensorFlow).
    Example: Store weights as a simple binary file (.mlx) and load it efficiently.

💡 Step 4: Benchmark & Optimize

    Run tests on Raspberry Pi, ESP32, low-power CPUs to measure speed & RAM usage.
    Optimize hot loops with SIMD intrinsics.
