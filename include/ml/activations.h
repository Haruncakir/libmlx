#ifndef MLX_ACTIVATIONS_H
#define MLX_ACTIVATIONS_H

#ifdef _MSC_VER
    typedef unsigned __int32 uint32_t;
#else
    typedef unsigned int __attribute__((mode(SI))) uint32_t;
#endif

uint32_t seed = 123456789; // Initial seed value

// Function to set the seed
void mlxsetseedrand(uint32_t new_seed) {
    seed = new_seed;
}

// Function to generate the next random number using inline assembly
unsigned int mlxlcgrand() {
    uint32_t a = 1664525; // Multiplier
    uint32_t c = 1013904223; // Increment
    uint32_t m = 0xFFFFFFFF; // Modulus (2^32)

    uint32_t next_seed;

    // Inline assembly to perform the LCG calculation
    asm (
        "mov %[seed], %%eax;"        // Load seed into EAX
        "imul %[a], %%eax;"          // EAX = EAX * a
        "add %[c], %%eax;"           // EAX = EAX + c
        "mov %%eax, %[next_seed];"   // Store result in next_seed
        "and %[m], %[next_seed];"    // Apply modulus
        : [next_seed] "=r" (next_seed) // Output
        : [seed] "r" (seed), [a] "r" (a), [c] "r" (c), [m] "r" (m) // Inputs
        : "%eax" // Clobbered register
    );

    seed = next_seed; // Update the seed
    return next_seed; // Return the random number
}

/**
 * @brief Custom implementation of exponential function
 * @param x Input value
 * @return Approximation of e^x
 */

static inline float mlxmatexpf(float x) {
    // Handle special cases for extreme values
    if (x >= 88.0f) return 3.4028235e+38f; // Near float max value
    if (x <= -88.0f) return 0.0f;           // Near zero
    if (x == 0.0f) return 1.0f;             // e^0 = 1

    // Use the identity e^x = e^(int_part + frac_part) = e^int_part * e^frac_part
    int int_part = (int)x;                  // Integer part of x
    float frac_part = x - int_part;         // Fractional part of x

    // Compute e^int_part using repeated multiplication
    float result = 1.0f;
    float e = 2.718281828459f;              // Value of e
    if (int_part > 0) {
        for (int i = 0; i < int_part; i++) {
            result *= e;
        }
    } else if (int_part < 0) {
        for (int i = 0; i < -int_part; i++) {
            result /= e;
        }
    }

    // Compute e^frac_part using a fast polynomial approximation
    // This is a 4th-order polynomial approximation for e^x over the range [-1, 1]
    float x2 = frac_part * frac_part;
    float x3 = x2 * frac_part;
    float x4 = x3 * frac_part;
    float poly = 1.0f + frac_part + 0.5f * x2 + 0.1666667f * x3 + 0.0416667f * x4;

    // Multiply the results
    return result * poly;
}

/**
 * @brief Custom implementation of natural logarithm
 * @param x Input value
 * @return Approximation of ln(x)
 */
float mlxmatlogf(float x) {
    // Handle special cases
    if (x <= 0.0f) return -3.4028235e+38f; // Return negative max float for invalid input
    if (x == 1.0f) return 0.0f;
    
    // Extract exponent using bit manipulation
    union {
        float f;
        uint32_t i;
    } u;
    u.f = x;
    
    int exp = ((u.i >> 23) & 0xFF) - 127;
    
    // Normalize x to [1, 2)
    u.i = (u.i & 0x807FFFFF) | 0x3F800000;
    float normalized = u.f;
    
    // Use a polynomial approximation for log(normalized) in [1, 2)
    // This is a minimax polynomial approximation
    float y = normalized - 1.0f;
    float y2 = y * y;
    float y4 = y2 * y2;
    
    float p = -0.33333333333f;
    p = p * y + 0.5f;
    p = p * y - 1.0f;
    p = p * y + 1.0f;
    
    return y - y2 * 0.5f + y4 * p + exp * 0.693147180559f; // 0.693... is ln(2)
}

/**
 * @brief Custom implementation of hyperbolic tangent
 * @param x Input value
 * @return Approximation of tanh(x)
 */
static inline float mlxmattanhf(float x) {
    // Handle special cases
    if (x >= 5.0f) return 1.0f;      // tanh approaches 1 for large x
    if (x <= -5.0f) return -1.0f;    // tanh approaches -1 for large negative x
    
    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    // For better numerical stability:
    // tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x)) for x > 0
    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1) for x < 0
    
    if (x >= 0.0f) {
        float exp_neg_2x = mlxmatexpf(-2.0f * x);
        return (1.0f - exp_neg_2x) / (1.0f + exp_neg_2x);
    } else {
        float exp_2x = mlxmatexpf(2.0f * x);
        return (exp_2x - 1.0f) / (exp_2x + 1.0f);
    }
}


/**
 * @brief ReLU (Rectified Linear Unit) activation function
 * @param x Input value
 * @return max(0, x)
 */
static inline float mlxactfrelu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * @brief Leaky ReLU activation function
 * @param x Input value
 * @param alpha Slope for negative values (typically 0.01)
 * @return x if x > 0, alpha * x otherwise
 */
static inline float mlxactfleakyrelu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

/**
 * @brief Parametric ReLU activation function
 * @param x Input value
 * @param alpha Learned slope parameter for negative values
 * @return x if x > 0, alpha * x otherwise
 */
static inline float mlxactfprelu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

/**
 * @brief ELU (Exponential Linear Unit) activation function
 * @param x Input value
 * @param alpha Scale for the exponential part (typically 1.0)
 * @return x if x > 0, alpha * (exp(x) - 1) otherwise
 */
static inline float mlxactfelu(float x, float alpha) {
    return x > 0.0f ? x : alpha * (mlxmatexpf(x) - 1.0f);
}

/**
 * @brief SELU (Scaled Exponential Linear Unit) activation function
 * @param x Input value
 * @return scale * x if x > 0, scale * alpha * (exp(x) - 1) otherwise
 */
static inline float mlxactfselu(float x) {
    const float alpha = 1.67326324f;
    const float scale = 1.05070098f;
    return x > 0.0f ? scale * x : scale * alpha * (mlxmatexpf(x) - 1.0f);
}

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return 1 / (1 + exp(-x))
 */
static inline float mlxactfsigmoid(float x) {
    return 1.0f / (1.0f + mlxmatexpf(-x));
}

/**
 * @brief Hard Sigmoid activation function (faster approximation of sigmoid)
 * @param x Input value
 * @return 0 if x < -2.5, 1 if x > 2.5, 0.2 * x + 0.5 otherwise
 */
static inline float mlxactfhardsigmoid(float x) {
    if (x < -2.5f) return 0.0f;
    if (x > 2.5f) return 1.0f;
    return 0.2f * x + 0.5f;
}

/**
 * @brief Tanh activation function
 * @param x Input value
 * @return tanh(x)
 */
static inline float mlxactftanh(float x) {
    return mlxmattanhf(x);
}

/**
 * @brief Softplus activation function
 * @param x Input value
 * @return log(1 + exp(x))
 */
static inline float mlxactfsoftplus(float x) {
    // For numerical stability, use different computation for large x
    if (x > 15.0f) return x; // If x is large, softplus(x) ≈ x
    return mlxmatlogf(1.0f + mlxmatexpf(x));
}

/**
 * @brief Swish activation function
 * @param x Input value
 * @param beta Trainable parameter (defaults to 1.0)
 * @return x * sigmoid(beta * x)
 */
static inline float mlxactfswish(float x, float beta) {
    return x * (1.0f / (1.0f + mlxmatexpf(-beta * x)));
}

/**
 * @brief GELU (Gaussian Error Linear Unit) activation function
 * @param x Input value
 * @return x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
static inline float mlxactfgelu(float x) {
    // Constants for GELU approximation
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    
    // Approximation of GELU using tanh
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + mlxmattanhf(inner));
}

/**
 * @brief Softmax activation function (for array of values)
 * @param x Array of input values
 * @param result Pre-allocated array to store results
 * @param size Size of the arrays
 */
static inline void mlxactfsoftmax(const float* x, float* result, int size) {
    if (size <= 0) return;
    
    // Find maximum value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        result[i] = mlxmatexpf(x[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    if (sum != 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            result[i] *= inv_sum;
        }
    }
}

/**
 * @brief Mish activation function
 * @param x Input value
 * @return x * tanh(softplus(x))
 */
static inline float mlxactfmish(float x) {
    // Softplus calculation
    float sp = x > 15.0f ? x : mlxmatlogf(1.0f + mlxmatexpf(x));
    // return x * tanh(softplus(x))
    return x * mlxmattanhf(sp);
}

#endif /* MLX_ACTIVATIONS_H */