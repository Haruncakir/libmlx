#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../include/ml/logistic.h"

// Small example dataset: 2D points with binary labels
// Points above the line y = x are class 1, below are class 0
#define NUM_SAMPLES 100
#define NUM_FEATURES 2

// Function to generate synthetic data
void generate_data(float* X, float* y, size_t n_samples) {
    // Seed random number generator
    srand(time(NULL));
    
    size_t i = 0;
    for (; i < n_samples * .8; ++i) {
        // Generate random points between -1 and 1
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        
        // Store features
        X[i * NUM_FEATURES] = x1;
        X[i * NUM_FEATURES + 1] = x2;
        
        // Assign label: 1 if x2 > x1, 0 otherwise (separable by line y = x)
        y[i] = (x2 > x1) ? 1.0f : 0.0f;
    }
    // noise
    for (;i < n_samples; ++i) {
        // Generate random points between -1 and 1
        float x1 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float x2 = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        
        // Store features
        X[i * NUM_FEATURES] = x1;
        X[i * NUM_FEATURES + 1] = x2;
        
        // Assign label: 0 if x2 > x1, 1 otherwise
        y[i] = (x2 > x1) ? 0.0f : 1.0f;
    }
}

// Function to evaluate model accuracy
float compute_accuracy(float* predictions, float* targets, size_t n_samples) {
    size_t correct = 0;
    
    for (size_t i = 0; i < n_samples; i++) {
        if (predictions[i] == targets[i]) {
            correct++;
        }
    }
    
    return (float)correct / n_samples;
}

// Function to get memory usage in MB (platform-specific)
float get_memory_usage_mb() {
    // This is a placeholder - actual implementation depends on your platform
    // For Linux, you could parse /proc/self/statm
    // For Windows, you could use GetProcessMemoryInfo()
    // For simplicity, we'll return a placeholder value
    // Replace this with actual memory measurement code for your platform
    FILE* file = fopen("/proc/self/statm", "r");
    if (file == NULL) {
        return -1.0f; // Error reading file
    }
    
    unsigned long vm_size;
    if (fscanf(file, "%lu", &vm_size) != 1) {
        fclose(file);
        return -1.0f;
    }
    
    fclose(file);
    // Convert to MB (page size is typically 4KB)
    return (float)(vm_size * 4.0f / 1024.0f);
}

int main() {
    printf("Logistic Regression Example - Binary Classification\n");
    printf("---------------------------------------------------\n");
    
    // Track initial memory usage
    float initial_memory = get_memory_usage_mb();
    
    // 1. Allocate memory region for the model and operations
    size_t region_size = 1024 * 512;
    unsigned char* memory[region_size];
    
    mat_region_t region;
    mat_status_t mat_status = reginit(&region, memory, region_size);
    if (mat_status != MATRIX_SUCCESS) {
        printf("Failed to initialize memory region: %s\n", strmaterr(mat_status));
        return 1;
    }
    
    // 2. Generate synthetic data
    float X_data[NUM_SAMPLES * NUM_FEATURES];
    float y_data[NUM_SAMPLES];
    
    generate_data(X_data, y_data, NUM_SAMPLES);

    FILE *data_file = fopen("./data/data_file.csv", "w");
    fprintf(data_file, "X1,X2,Y\n");
    for (size_t i = 0; i < NUM_SAMPLES; ++i)
        fprintf(data_file, "%f,%f,%f\n", X_data[i * NUM_FEATURES], X_data[i * NUM_FEATURES + 1], y_data[i]);
    fclose(data_file);

    // 3. Create matrix for features
    mat_t X;
    mat_status = matcreate(&region, NUM_SAMPLES, NUM_FEATURES, X_data, &X);
    if (mat_status != MATRIX_SUCCESS) {
        printf("Failed to create feature matrix: %s\n", strmaterr(mat_status));
        return 1;
    }
    
    // 4. Configure and initialize logistic regression model
    mlxlogistic_config_t config;
    mlxlogregconfiginit(&config);
    
    // Customize configuration
    config.learning_rate = 0.01f;
    config.l2_regularization = 0.1f;
    config.max_iterations = 5000; // 1000
    config.convergence_tol = 1e-6f;
    config.fit_intercept = mlxbooltrue;
    config.verbose = mlxbooltrue;
    
    mlxlogistic_model_t model;
    mlxlogistic_status_t status = mlxlogreginit(&model, &region, NUM_FEATURES, 1, &config);
    
    if (status != LOGISTIC_SUCCESS) {
        printf("Failed to initialize model: %s\n", mlxlogregstrerror(status));
        return 1;
    }
    
    // Memory after initialization
    float mem_after_init = get_memory_usage_mb();
    
    // 5. Train the model
    printf("\nTraining logistic regression model...\n");
    
    clock_t start = clock();
    status = mlxlogregtrain(&model, &X, y_data, &region);
    clock_t end = clock();
    
    if (status != LOGISTIC_SUCCESS) {
        printf("Training failed: %s\n", mlxlogregstrerror(status));
        return 1;
    }
    
    double train_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Training completed in %.3f seconds\n\n", train_time);
    
    // Memory after training
    float mem_after_training = get_memory_usage_mb();
    
    // 6. Print model coefficients
    printf("Model coefficients:\n");
    if (config.fit_intercept) {
        printf("  Intercept: %.6f\n", model.weights.data[0]);
        for (size_t i = 1; i < model.weights.col; i++) {
            printf("  Feature %zu: %.6f\n", i-1, model.weights.data[i]);
        }
    } else {
        for (size_t i = 0; i < model.weights.col; i++) {
            printf("  Feature %zu: %.6f\n", i, model.weights.data[i]);
        }
    }
    printf("\n");

    printf("Mat count: %zu, region used: %zu\n", region.mat_count, region.used);
    float memory_usage = (float)region.used / (1024.0f * 1024.0f); // Convert to MB
    printf("Memory used by matrix operations: %.2f MB\n", memory_usage);
    
    // 7. Make predictions
    float predictions[NUM_SAMPLES];
    status = mlxlogregpredict(&model, &X, predictions, 0.5, &region);
    
    if (status != LOGISTIC_SUCCESS) {
        printf("Prediction failed: %s\n", mlxlogregstrerror(status));
        return 1;
    }
    
    // 8. Evaluate model
    float accuracy = compute_accuracy(predictions, y_data, NUM_SAMPLES);
    printf("Model accuracy: %.2f%%\n", accuracy * 100.0f);
    
    // 9. Calculate loss
    float loss;
    status = mlxlogregcrossentropy(&model, &X, y_data, &loss, &region);
    
    if (status == LOGISTIC_SUCCESS) {
        printf("Log loss: %.6f\n", loss);
    }
    
    // Memory after prediction
    float mem_after_prediction = get_memory_usage_mb();
    
    // 10. Get predicted probabilities
    float probabilities[NUM_SAMPLES];
    status = mlxlogregpredictproba(&model, &X, probabilities, &region);
    
    // Print a few predictions with probabilities
    printf("\nSample predictions:\n");
    printf("  Index | Features            | True Label | Prediction | Probability\n");
    printf("  ------|---------------------|------------|------------|------------\n");

    for (size_t i = 0; i < 10; i++) {
        printf("  %4zu  | (%7.4f, %7.4f)  | %10d | %10d | %11.6f\n",
            i, X_data[i*NUM_FEATURES], X_data[i*NUM_FEATURES+1],
            (int)y_data[i], (int)predictions[i], probabilities[i]);
    }
    
    // Save metrics to a file for Python to read
    FILE *metrics_file = fopen("./data/c_model_metrics.csv", "w");
    fprintf(metrics_file, "metric,value\n");
    fprintf(metrics_file, "training_time,%.6f\n", train_time);
    fprintf(metrics_file, "memory_usage_mb,%.2f\n", memory_usage);
    fprintf(metrics_file, "memory_init_mb,%.2f\n", mem_after_init - initial_memory);
    fprintf(metrics_file, "memory_training_mb,%.2f\n", mem_after_training - mem_after_init);
    fprintf(metrics_file, "memory_prediction_mb,%.2f\n", mem_after_prediction - mem_after_training);
    fprintf(metrics_file, "total_memory_mb,%.2f\n", mem_after_prediction - initial_memory);
    fprintf(metrics_file, "accuracy,%.6f\n", accuracy);
    fprintf(metrics_file, "log_loss,%.6f\n", loss);
    fprintf(metrics_file, "iterations,%d\n", model.config.max_iterations);
    fprintf(metrics_file, "intercept,%.6f\n", config.fit_intercept ? model.weights.data[0] : 0.0f);
    fprintf(metrics_file, "weight_1,%.6f\n", config.fit_intercept ? model.weights.data[1] : model.weights.data[0]);
    fprintf(metrics_file, "weight_2,%.6f\n", config.fit_intercept ? model.weights.data[2] : model.weights.data[1]);
    fclose(metrics_file);
    
    printf("\nMetrics saved to ./data/c_model_metrics.csv\n");
    
    regreset(&region);
    return 0;
}