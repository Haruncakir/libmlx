// Example usage of the logistic regression library with visual representations

#include "../../include/matrix/matpool.h"
#include "../../include/matrix/matrix_config.h"
#include "../../include/ml/logistic.h"

int main() {
    // Create a memory region for our matrices and model
    unsigned char memory[10240];
    mat_region_t region;
    reginit(&region, memory, sizeof(memory));
    
    /*
     * PART 1: BINARY LOGISTIC REGRESSION EXAMPLE
     */
    
    /* 
     * mlxlogistic_config_t Visualization:
     * 
     * struct {
     *   learning_rate: 0.01,      // Controls step size in gradient descent
     *   l2_regularization: 0.1,   // Prevents overfitting: larger values = more regularization
     *   max_iterations: 100,      // Maximum training iterations
     *   convergence_tol: 0.0001,  // Stop when changes are smaller than this
     *   fit_intercept: true,      // Add bias term (x_0 = 1)
     *   verbose: true             // Print training progress
     * }
     */
    mlxlogistic_config_t config;
    mlxlogregconfiginit(&config);
    config.learning_rate = 0.01f;
    config.l2_regularization = 0.1f;
    config.max_iterations = 100;
    config.convergence_tol = 0.0001f;
    config.fit_intercept = true;
    config.verbose = true;
    
    // Initialize a binary logistic regression model (num_classes = 1)
    mlxlogistic_model_t binary_model;
    size_t num_features = 2;  // 2 features for simplicity
    
    mlxlogreginit(&binary_model, &region, num_features, 1, &config);
    
    /* 
     * mlxlogistic_model_t Visualization for Binary Classification:
     * 
     * binary_model = {
     *   weights: {               // Matrix of shape (1, 3) for binary classification with intercept
     *     row: 1,                // One row for binary classification
     *     col: 3,                // num_features + 1 for intercept
     *     stride: 8,             // Aligned for SIMD operations (may vary)
     *     data: [0.0, 0.0, 0.0]  // Initially zeros, will be learned during training
     *   },
     *   has_intercept: true,     // We're using an intercept term
     *   num_features: 2,         // Original number of features (excluding intercept)
     *   num_classes: 1,          // Binary classification
     *   workspace: (pointer),    // Temporary calculation space
     *   workspace_size: X,       // Size of workspace
     *   config: {...}            // Configuration copied from above
     * }
     * 
     * The weights represent: [intercept, weight_feature1, weight_feature2]
     */
    
    // Create a sample dataset: 4 samples, 2 features
    // Example: diabetes prediction based on glucose level and BMI
    mat_t X;
    float X_data[] = {
        85.0f, 24.5f,  // Sample 1: glucose=85, BMI=24.5
        155.0f, 32.1f, // Sample 2: glucose=155, BMI=32.1
        120.0f, 29.8f, // Sample 3: glucose=120, BMI=29.8
        95.0f, 23.0f   // Sample 4: glucose=95, BMI=23.0
    };
    matcreate(&region, 4, 2, X_data, &X);
    
    /* 
     * X Matrix (Features) Visualization:
     * Logical view (4x2):
     * [  85.0,  24.5 ]  <- Sample 1: glucose=85, BMI=24.5
     * [ 155.0,  32.1 ]  <- Sample 2: glucose=155, BMI=32.1
     * [ 120.0,  29.8 ]  <- Sample 3: glucose=120, BMI=29.8
     * [  95.0,  23.0 ]  <- Sample 4: glucose=95, BMI=23.0
     * 
     * Physical layout with stride (example if SIMD_ALIGN=32 bytes):
     * [ 85.0, 24.5, pad, pad, pad, pad, pad, pad ]
     * [ 155.0, 32.1, pad, pad, pad, pad, pad, pad ]
     * [ 120.0, 29.8, pad, pad, pad, pad, pad, pad ]
     * [ 95.0, 23.0, pad, pad, pad, pad, pad, pad ]
     */
    
    // Target labels (0: no diabetes, 1: diabetes)
    float y[] = {
        0.0f,  // Sample 1: no diabetes
        1.0f,  // Sample 2: has diabetes
        1.0f,  // Sample 3: has diabetes
        0.0f   // Sample 4: no diabetes
    };
    
    /*
     * Target vector y Visualization:
     * [0.0, 1.0, 1.0, 0.0]
     */
    
    // Train the binary logistic regression model
    mlxlogregtrain(&binary_model, &X, y, &region);
    
    /* 
     * Training Process Visualization (Binary Logistic Regression):
     * 
     * 1. For each iteration:
     *    a. Compute linear scores: z = X * weights^T
     *       For sample 1 with intercept: z = 1 * w_0 + 85.0 * w_1 + 24.5 * w_2
     * 
     *    b. Apply sigmoid function: p = 1 / (1 + exp(-z))
     *       This gives probability of class 1 (range: 0 to 1)
     * 
     *    c. Compute error: error = p - y
     *       Example: If p=0.7 and y=1, error = -0.3
     * 
     *    d. Compute gradients: grad = (X^T * error) / num_samples + regularization
     *       - For intercept: grad_0 = mean(error)
     *       - For feature j: grad_j = mean(X_j * error) + 2 * lambda * w_j
     * 
     *    e. Update weights: weights = weights - learning_rate * grad
     * 
     *    f. Check convergence: if weight changes < tolerance, stop
     * 
     * 2. After training, binary_model.weights might look like:
     *    weights = [-4.2, 0.035, 0.12]
     *    - Negative intercept (-4.2) means baseline tendency to predict class 0
     *    - Positive weight for glucose (0.035) means higher glucose increases diabetes risk
     *    - Positive weight for BMI (0.12) means higher BMI increases diabetes risk
     */
    
    // Predict probabilities for new samples
    float probs[4];
    mlxlogregpredictproba(&binary_model, &X, probs, &region);
    
    /* 
     * Prediction Process Visualization (Binary Logistic Regression):
     * 
     * 1. Compute linear scores (z) as above
     * 2. Apply sigmoid function: p = 1 / (1 + exp(-z))
     * 
     * Example if weights = [-4.2, 0.035, 0.12]:
     * 
     * Sample 1: z = -4.2 + 0.035*85.0 + 0.12*24.5 = -4.2 + 2.975 + 2.94 = 1.715
     *           p = 1 / (1 + exp(-1.715)) = 0.847
     * 
     * Sample 2: z = -4.2 + 0.035*155.0 + 0.12*32.1 = -4.2 + 5.425 + 3.852 = 5.077
     *           p = 1 / (1 + exp(-5.077)) = 0.994
     * 
     * Final probs array would be about:
     * probs = [0.178, 0.994, 0.966, 0.234]
     * 
     * This means:
     * - Sample 1: 17.8% probability of diabetes
     * - Sample 2: 99.4% probability of diabetes
     * - Sample 3: 96.6% probability of diabetes
     * - Sample 4: 23.4% probability of diabetes
     */
    
    // Predict class labels
    float labels[4];
    mlxlogregpredict(&binary_model, &X, labels, &region);
    
    /* 
     * Class Prediction Visualization:
     * 
     * For binary classification: 
     * - If probability > 0.5, predict class 1
     * - Otherwise, predict class 0
     * 
     * labels = [0.0, 1.0, 1.0, 0.0]
     */
    
    // Calculate cross-entropy loss
    float loss;
    mlxlogregcrossentropy(&binary_model, &X, y, &loss, &region);
    
    /* 
     * Cross-Entropy Loss Calculation Visualization:
     * 
     * For binary classification:
     * Loss = -1/n * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
     * 
     * Sample 1: y=0, p=0.178  => -(0*log(0.178) + 1*log(1-0.178)) = -log(0.822) = 0.196
     * Sample 2: y=1, p=0.994  => -(1*log(0.994) + 0*log(1-0.994)) = -log(0.994) = 0.006
     * Sample 3: y=1, p=0.966  => -(1*log(0.966) + 0*log(1-0.966)) = -log(0.966) = 0.035
     * Sample 4: y=0, p=0.234  => -(0*log(0.234) + 1*log(1-0.234)) = -log(0.766) = 0.267
     * 
     * Average loss = (0.196 + 0.006 + 0.035 + 0.267) / 4 = 0.126
     * 
     * Plus L2 regularization term: lambda * sum(w_j^2) / (2*n)
     */
    
    /*
     * PART 2: MULTINOMIAL LOGISTIC REGRESSION EXAMPLE (3 classes)
     */
    
    // Reset the memory region
    regreset(&region);
    
    // Initialize a multinomial logistic regression model (num_classes = 3)
    // For example: iris classification (setosa, versicolor, virginica)
    mlxlogistic_model_t multi_model;
    num_features = 4;  // 4 features for iris: sepal length, sepal width, petal length, petal width
    
    mlxlogreginit(&multi_model, &region, num_features, 3, &config);
    
    /* 
     * mlxlogistic_model_t Visualization for Multinomial Classification:
     * 
     * multi_model = {
     *   weights: {                     // Matrix of shape (3, 5) for 3 classes with intercept
     *     row: 3,                      // One row per class
     *     col: 5,                      // num_features + 1 for intercept
     *     stride: 8,                   // Aligned for SIMD (may vary)
     *     data: [                      // 3 rows, 5 columns
     *       [0.0, 0.0, 0.0, 0.0, 0.0], // Class 0 (setosa) weights
     *       [0.0, 0.0, 0.0, 0.0, 0.0], // Class 1 (versicolor) weights
     *       [0.0, 0.0, 0.0, 0.0, 0.0]  // Class 2 (virginica) weights
     *     ]
     *   },
     *   has_intercept: true,          // Using intercept term
     *   num_features: 4,              // Original features (excluding intercept)
     *   num_classes: 3,               // 3 output classes
     *   workspace: (pointer),         // Temporary calculation space
     *   workspace_size: X,            // Size of workspace
     *   config: {...}                 // Configuration
     * }
     * 
     * Each row of weights represents:
     * [intercept, w_sepal_length, w_sepal_width, w_petal_length, w_petal_width]
     */
    
    // Create a sample iris dataset: 6 samples, 4 features
    mat_t X_iris;
    float X_iris_data[] = {
        // Format: sepal_length, sepal_width, petal_length, petal_width
        5.1f, 3.5f, 1.4f, 0.2f,  // Sample 1: typical setosa
        7.0f, 3.2f, 4.7f, 1.4f,  // Sample 2: typical versicolor
        6.3f, 3.3f, 6.0f, 2.5f,  // Sample 3: typical virginica
        4.9f, 3.0f, 1.4f, 0.2f,  // Sample 4: another setosa
        5.9f, 3.0f, 4.2f, 1.5f,  // Sample 5: another versicolor
        6.5f, 3.0f, 5.8f, 2.2f   // Sample 6: another virginica
    };
    matcreate(&region, 6, 4, X_iris_data, &X_iris);
    
    /* 
     * X_iris Matrix (Features) Visualization:
     * Logical view (6x4):
     * [ 5.1, 3.5, 1.4, 0.2 ]  <- Sample 1: setosa
     * [ 7.0, 3.2, 4.7, 1.4 ]  <- Sample 2: versicolor
     * [ 6.3, 3.3, 6.0, 2.5 ]  <- Sample 3: virginica
     * [ 4.9, 3.0, 1.4, 0.2 ]  <- Sample 4: setosa
     * [ 5.9, 3.0, 4.2, 1.5 ]  <- Sample 5: versicolor
     * [ 6.5, 3.0, 5.8, 2.2 ]  <- Sample 6: virginica
     */
    
    // Target labels (0: setosa, 1: versicolor, 2: virginica)
    float y_iris[] = {
        0.0f,  // Sample 1: setosa
        1.0f,  // Sample 2: versicolor
        2.0f,  // Sample 3: virginica
        0.0f,  // Sample 4: setosa
        1.0f,  // Sample 5: versicolor
        2.0f   // Sample 6: virginica
    };
    
    /*
     * Target vector y_iris Visualization:
     * [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
     */
    
    // Train the multinomial logistic regression model
    mlxlogregtrain(&multi_model, &X_iris, y_iris, &region);
    
    /* 
     * Training Process Visualization (Multinomial Logistic Regression):
     * 
     * 1. For each iteration:
     *    a. Compute linear scores for each class: Z = X * weights^T
     *       Z is a matrix of shape (n_samples, n_classes)
     *       For sample 1, class 0: 
     *         z_10 = 1 * w_00 + 5.1 * w_01 + 3.5 * w_02 + 1.4 * w_03 + 0.2 * w_04
     * 
     *    b. Apply softmax function to get probabilities:
     *       p_ij = exp(z_ij) / sum_k(exp(z_ik))
     *       This normalizes scores to probabilities that sum to 1 across classes
     * 
     *    c. Compute error: error_ij = (y_i == j) - p_ij
     *       Error matrix has shape (n_samples, n_classes)
     * 
     *    d. Compute gradients: grad = X^T * error + regularization
     * 
     *    e. Update weights: weights = weights - learning_rate * grad
     * 
     *    f. Check convergence
     * 
     * 2. After training, multi_model.weights might look like:
     *    Class 0 (setosa): [5.0, -0.5, 1.8, -2.9, -3.1]
     *    Class 1 (versicolor): [2.0, -0.2, 0.3, 0.8, 0.7]
     *    Class 2 (virginica): [-7.0, 0.7, -2.1, 2.1, 2.4]
     * 
     *    Interpreting these weights:
     *    - Large negative weight for petal length (-2.9) in setosa class means 
     *      shorter petals strongly indicate setosa
     *    - Large positive weight for petal width (2.4) in virginica class means
     *      wider petals strongly indicate virginica
     */
    
    // Predict probabilities for the samples
    float probs_multi[6 * 3];  // 6 samples, 3 classes each
    mlxlogregpredictproba(&multi_model, &X_iris, probs_multi, &region);
    
    /* 
     * Prediction Process Visualization (Multinomial Logistic Regression):
     * 
     * 1. Compute linear scores (z) for each class
     * 2. Apply softmax function to get normalized probabilities
     * 
     * Example output might look like:
     * 
     * Sample 1 (setosa):     [0.98, 0.02, 0.00]  <- 98% setosa, 2% versicolor, ~0% virginica
     * Sample 2 (versicolor): [0.01, 0.85, 0.14]  <- 1% setosa, 85% versicolor, 14% virginica
     * Sample 3 (virginica):  [0.00, 0.05, 0.95]  <- ~0% setosa, 5% versicolor, 95% virginica
     * Sample 4 (setosa):     [0.97, 0.03, 0.00]  <- 97% setosa, 3% versicolor, ~0% virginica
     * Sample 5 (versicolor): [0.03, 0.92, 0.05]  <- 3% setosa, 92% versicolor, 5% virginica
     * Sample 6 (virginica):  [0.00, 0.08, 0.92]  <- ~0% setosa, 8% versicolor, 92% virginica
     * 
     * In memory, this would be stored as a flat array:
     * probs_multi = [0.98, 0.02, 0.00, 0.01, 0.85, 0.14, ...]
     */
    
    // Predict class labels
    float labels_multi[6];
    mlxlogregpredict(&multi_model, &X_iris, labels_multi, &region);
    
    /* 
     * Class Prediction Visualization (Multinomial):
     * 
     * For multinomial classification:
     * - Predict the class with highest probability
     * 
     * labels_multi = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
     * 
     * Meaning:
     * Sample 1: predicted as class 0 (setosa)
     * Sample 2: predicted as class 1 (versicolor)
     * ...and so on
     */
    
    // Calculate cross-entropy loss
    float loss_multi;
    mlxlogregcrossentropy(&multi_model, &X_iris, y_iris, &loss_multi, &region);
    
    /* 
     * Cross-Entropy Loss Calculation Visualization (Multinomial):
     * 
     * For multinomial classification:
     * Loss = -1/n * sum(sum(I(y_i == j) * log(p_ij)))
     * 
     * Where I(y_i == j) is 1 if y_i equals j, and 0 otherwise
     * 
     * Sample 1 (y=0): -log(0.98) = 0.02
     * Sample 2 (y=1): -log(0.85) = 0.16
     * Sample 3 (y=2): -log(0.95) = 0.05
     * Sample 4 (y=0): -log(0.97) = 0.03
     * Sample 5 (y=1): -log(0.92) = 0.08
     * Sample 6 (y=2): -log(0.92) = 0.08
     * 
     * Average loss = (0.02 + 0.16 + 0.05 + 0.03 + 0.08 + 0.08) / 6 = 0.07
     * 
     * Plus L2 regularization term
     */
    
    return 0;
}