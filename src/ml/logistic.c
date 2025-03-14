#include "../../include/ml/logistic.h"
#include "../../include/matrix/matpool.h"

logistic_status_t mlxlogregconfiginit(logistic_config_t* config) {
    if (!config) {
        return LOGISTIC_NULL_POINTER;
    }
    
    // Set default values
    config->learning_rate = 0.1f;
    config->l2_regularization = 0.0f;
    config->max_iterations = 100;
    config->convergence_tol = 1e-4f;
    config->fit_intercept = true;
    config->verbose = false;
    
    return LOGISTIC_SUCCESS;
}

logistic_status_t mlxlogreginit(logistic_model_t* model, 
                               mat_region_t* reg,
                               size_t num_features, 
                               size_t num_classes,
                               const logistic_config_t* config) {
                               
    if (!model || !reg) {
        return LOGISTIC_NULL_POINTER;
    }
    
    if (num_features == 0 || num_classes == 0) {
        return LOGISTIC_INVALID_PARAMETER;
    }
    
    // Apply default configuration if none provided
    logistic_config_t default_config;
    if (!config) {
        logistic_config_init(&default_config);
        config = &default_config;
    }
    
    // Copy configuration
    model->config = *config;
    
    // Set model metadata
    model->num_features = num_features;
    model->num_classes = num_classes;
    model->has_intercept = config->fit_intercept;
    
    // Calculate actual feature count (include intercept if needed)
    size_t actual_features = num_features + (config->fit_intercept ? 1 : 0);
    
    // For binary classification, we need only one set of weights
    // For multiclass, we need one set per class
    size_t weight_rows = (num_classes > 1) ? num_classes : 1;
    
    // Allocate weight matrix
    mat_status_t status = matalloc(reg, weight_rows, actual_features, &model->weights);
    if (status != MATRIX_SUCCESS) {
        return LOGISTIC_MEMORY_ERROR;
    }
    
    // Initialize weights to zero
    for (size_t i = 0; i < model->weights.row; i++) {
        for (size_t j = 0; j < model->weights.col; j++) {
            model->weights.data[i * model->weights.stride + j] = 0.0f;
        }
    }
    
    // Allocate workspace for computations during prediction
    // We need space for the largest intermediate result during prediction
    // For binary: one vector of size max(num_samples)
    // For multiclass: num_classes * num_samples
    model->workspace_size = (num_classes > 1) ? (num_classes * 256) : 256;
    /* TODO: regalloc optimizes for matrix specific operations. Requires different memory management. */
    model->workspace = (float*)regalloc(reg, model->workspace_size * sizeof(float));
    
    if (!model->workspace) {
        return LOGISTIC_MEMORY_ERROR;
    }
    
    return LOGISTIC_SUCCESS;
}
