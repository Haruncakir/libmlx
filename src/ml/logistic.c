#include "../../include/ml/logistic.h"
#include "../../include/ml/activations.h"
#include "../../include/matrix/matpool.h"

#ifdef DEBUG
#include <stdio.h>
#endif

static void compute_sigmoid(const float* input, float* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (input[i] < -40.0f) {
            // Numerical stability for large negative values
            output[i] = 0.0f;
        } else if (input[i] > 40.0f) {
            // Numerical stability for large positive values
            output[i] = 1.0f;
        } else {
            output[i] = mlxactfsigmoid(input[i]);
        }
    }
}

mlxlogistic_status_t mlxlogregconfiginit(mlxlogistic_config_t* config) {
    if (!config) {
        return LOGISTIC_NULL_POINTER;
    }
    
    // Set default values
    config->learning_rate = 0.1f;
    config->l2_regularization = 0.0f;
    config->max_iterations = 100;
    config->convergence_tol = 1e-4f;
    config->fit_intercept = mlxbooltrue;
    config->verbose = mlxboolfalse;
    
    return LOGISTIC_SUCCESS;
}

mlxlogistic_status_t mlxlogreginit(mlxlogistic_model_t* model, 
                               mat_region_t* reg,
                               size_t num_features, 
                               size_t num_classes,
                               const mlxlogistic_config_t* config) {
                               
    if (!model || !reg) {
        return LOGISTIC_NULL_POINTER;
    }
    
    if (num_features == 0 || num_classes == 0) {
        return LOGISTIC_INVALID_PARAMETER;
    }
    
    // Apply default configuration if none provided
    mlxlogistic_config_t default_config;
    if (!config) {
        mlxlogregconfiginit(&default_config);
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

    // Allocate aligned workspace for SIMD operations
    mat_t workspace_mat;
    mat_status_t ws_status = matalloc(reg, 1, model->workspace_size, &workspace_mat);
    if (ws_status != MATRIX_SUCCESS) {
        return LOGISTIC_MEMORY_ERROR;
    }

    // Use the allocated matrix data as workspace
    model->workspace = workspace_mat.data; // can be dangerous
    
    if (!model->workspace) {
        return LOGISTIC_MEMORY_ERROR;
    }
    
    return LOGISTIC_SUCCESS;
}

static float __mlxfabsf(float x) {
    // Bit manipulation approach - clear the sign bit
    unsigned int* ptr = (unsigned int*)&x;
    *ptr &= 0x7FFFFFFF; // Mask out the sign bit (MSB)
    
    return x;
}


mlxlogistic_status_t mlxlogregtrain(mlxlogistic_model_t* model,
                                const mat_t* X,
                                const float* y,
                                mat_region_t* reg) {
    if (!model || !X || !y || !reg) {
        return LOGISTIC_NULL_POINTER;
    }
    
    // Check dimensions
    if (X->col != model->num_features) {
        return LOGISTIC_DIMENSION_MISMATCH;
    }
    
    // For binary classification
    if (model->num_classes <= 1) {
        size_t num_samples = X->row;
        size_t num_features = model->weights.col;
        
        // Allocate temporary matrices and vectors
        mat_t X_with_intercept;
        float* predictions;
        float* errors;
        float* gradients;
        
        // Prepare X with intercept if needed
        if (model->has_intercept) {
            // Allocate with space for the intercept column
            mat_status_t status = matalloc(reg, num_samples, num_features, &X_with_intercept);
            if (status != MATRIX_SUCCESS) {
                return LOGISTIC_MEMORY_ERROR;
            }
            
            // Copy features and add intercept column (set to 1.0)
            for (size_t i = 0; i < num_samples; i++) {
                // Set intercept
                X_with_intercept.data[i * X_with_intercept.stride] = 1.0f;
                
                // Copy original features
                for (size_t j = 0; j < X->col; j++) {
                    X_with_intercept.data[i * X_with_intercept.stride + j + 1] = 
                        X->data[i * X->stride + j];
                }
            }
        }
        
        // Get working matrix (either original or with intercept)
        const mat_t* X_work = model->has_intercept ? &X_with_intercept : X;

        mat_t predictions_mat;
        // mat_status_t ws_status = matalloc(reg, 1, num_samples * sizeof(float), &predictions_mat);
        mat_status_t ws_status = matalloc(reg, 1, num_samples, &predictions_mat);
        if (ws_status != MATRIX_SUCCESS) {
            return LOGISTIC_MEMORY_ERROR;
        }

        mat_t errors_mat;
        ws_status = matalloc(reg, 1, num_samples, &errors_mat);
        if (ws_status != MATRIX_SUCCESS) {
            return LOGISTIC_MEMORY_ERROR;
        }

        mat_t num_features_mat;
        ws_status = matalloc(reg, 1, num_samples, &num_features_mat);
        if (ws_status != MATRIX_SUCCESS) {
            return LOGISTIC_MEMORY_ERROR;
        }
        
        // Allocate vectors for computations
        predictions = predictions_mat.data;
        errors = errors_mat.data;
        gradients = num_features_mat.data;
        
        if (!predictions || !errors || !gradients) {
            return LOGISTIC_MEMORY_ERROR;
        }
        
        // Gradient descent loop
        float prev_loss = 1e30f;

        // 1. Compute predictions: sigmoid(X * weights)
        mat_t X_work_transpose;
        ws_status = matalloc(reg, X_work->col, X_work->row, &X_work_transpose);
        mattranspose(X_work, &X_work_transpose);
        
        for (size_t iter = 0; iter < model->config.max_iterations; ++iter) {
            for (size_t i = 0; i < num_samples; i++) {
                // Get pointer to the i-th row of X_work
                float *row = &X_work->data[i * X_work->stride];
                // Compute dot product with weights
                predictions[i] = matdot(row, model->weights.data, num_features);
            }

            //mat_status_t status = matvecmul(X_work, model->weights.data, predictions);
            //matvecmul(&predictions_mat, X_work->data, model->weights.data);
            //if (status != MATRIX_SUCCESS) {
            //    return LOGISTIC_MEMORY_ERROR;
            //}
            
            compute_sigmoid(predictions, predictions, num_samples);
            
            // 2. Compute errors: predictions - targets
            for (size_t i = 0; i < num_samples; i++) {
                errors[i] = predictions[i] - y[i];
            }
            
            // 3. Compute gradients: X^T * errors / num_samples + regularization
            for (size_t j = 0; j < num_features; j++) {
                float grad_sum = 0.0f;
                for (size_t i = 0; i < num_samples; i++) {
                    grad_sum += X_work->data[i * X_work->stride + j] * errors[i];
                }
                
                gradients[j] = grad_sum / num_samples;
                
                // Add L2 regularization (don't regularize intercept)
                if (model->config.l2_regularization > 0.0f && 
                    !(model->has_intercept && j == 0)) {
                    gradients[j] += model->config.l2_regularization * model->weights.data[j];
                }
            }
            
            // 4. Update weights: weights -= learning_rate * gradients
            for (size_t j = 0; j < num_features; j++) {
                model->weights.data[j] -= model->config.learning_rate * gradients[j];
            }
            
            // 5. Compute log loss for convergence check
            float loss = 0.0f;
            for (size_t i = 0; i < num_samples; i++) {
                float p = predictions[i];
                float t = y[i];
                
                // Clip probabilities for numerical stability
                if (p < 1e-15f) p = 1e-15f;
                if (p > 1.0f - 1e-15f) p = 1.0f - 1e-15f;
                
                loss -= t * mlxmatlogf(p) + (1.0f - t) * mlxmatlogf(1.0f - p);
            }
            loss /= num_samples;
#ifdef DEBUG
            printf("loss: %f\n", loss);
#endif
            
            // Add L2 regularization term to loss
            if (model->config.l2_regularization > 0.0f) {
                float reg_sum = 0.0f;
                size_t start_idx = model->has_intercept ? 1 : 0; // Skip intercept
                
                for (size_t j = start_idx; j < num_features; j++) {
                    reg_sum += model->weights.data[j] * model->weights.data[j];
                }
                
                loss += 0.5f * model->config.l2_regularization * reg_sum;
            }
            
            // Check for convergence
            if (__mlxfabsf(loss - prev_loss) < model->config.convergence_tol) {
                return LOGISTIC_SUCCESS;
            }
            
            prev_loss = loss;
        }
        
        // If we get here, we didn't converge within max_iterations
        return LOGISTIC_NOT_CONVERGED;
    }
    else {
        // Multi-class logistic regression (one-vs-rest for simplification)
        // A full implementation would use softmax
        return LOGISTIC_NOT_CONVERGED;  // Placeholder for multi-class
    }
}

mlxlogistic_status_t mlxlogregpredictproba(const mlxlogistic_model_t* model,
                                        const mat_t* X,
                                        float* probs,
                                        mat_region_t* reg) {
    if (!model || !X || !probs) {
        return LOGISTIC_NULL_POINTER;
    }
    
    // Check dimensions
    if (X->col != model->num_features) {
        return LOGISTIC_DIMENSION_MISMATCH;
    }
    
    size_t num_samples = X->row;
    
    // Handle binary classification
    if (model->num_classes <= 1) {
        // Create temporary matrix with intercept if needed
        mat_t X_with_intercept;
        
        if (model->has_intercept) {
            mat_status_t status = matalloc(reg, num_samples, model->weights.col, &X_with_intercept);
            if (status != MATRIX_SUCCESS) {
                return LOGISTIC_MEMORY_ERROR;
            }
            
            // Copy features and add intercept column
            for (size_t i = 0; i < num_samples; i++) {
                // Set intercept
                X_with_intercept.data[i * X_with_intercept.stride] = 1.0f;
                
                // Copy original features
                for (size_t j = 0; j < X->col; j++) {
                    X_with_intercept.data[i * X_with_intercept.stride + j + 1] = 
                        X->data[i * X->stride + j];
                }
            }
        }
        
        // Choose which matrix to use
        const mat_t* X_work = model->has_intercept ? &X_with_intercept : X;
        
        // Compute linear predictions: X * weights
        mat_t linear_preds_mat;
        if (matalloc(reg, 1, model->workspace_size, &linear_preds_mat) != MATRIX_SUCCESS) {
            return LOGISTIC_MEMORY_ERROR;
        }

        float* linear_preds = model->workspace;

/*
        size_t num_features = model->weights.col;
        for (size_t i = 0; i < num_samples; ++i) {
            // Get pointer to the i-th row of X_work
            float *row = &X_work->data[i * X_work->stride];
            // Compute dot product with weights
            linear_preds[i] = matdot(row, model->weights.data, num_features);
        }
*/
        // Compute predictions manually
        for (size_t i = 0; i < num_samples; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < model->weights.col; ++j) {
                sum += X_work->data[i * X_work->stride + j] * model->weights.data[j];
            }
            linear_preds[i] = sum;
        }

        /*
        mat_status_t status = matvecmul(&linear_preds_mat, X_work->data, model->weights.data);
        if (status != MATRIX_SUCCESS) {
            return LOGISTIC_MEMORY_ERROR;
        } */
        
        
        // Apply sigmoid to get probabilities
        compute_sigmoid(linear_preds, probs, num_samples);

        
        return LOGISTIC_SUCCESS;
    }
    else {
        // Multi-class classification
        // Placeholder for multi-class implementation
        return LOGISTIC_INVALID_PARAMETER;
    }
}

mlxlogistic_status_t mlxlogregpredict(const mlxlogistic_model_t* model,
                                  const mat_t* X,
                                  float* labels,
                                  float threshold,
                                  mat_region_t* reg) {
    if (!model || !X || !labels) {
        return LOGISTIC_NULL_POINTER;
    }
    
    size_t num_samples = X->row;
    
    // Get probabilities first
    float* probs = model->workspace;
    
    mlxlogistic_status_t status = mlxlogregpredictproba(model, X, probs, reg);
    if (status != LOGISTIC_SUCCESS) {
        return status;
    }
    
    // binary classification
    if (model->num_classes <= 1) {
        for (size_t i = 0; i < num_samples; i++) {
            labels[i] = (probs[i] >= threshold) ? 1.0f : 0.0f;
        }
    }
    else {
        // Multi-class: would select highest probability class
        // Placeholder for multi-class implementation
        return LOGISTIC_INVALID_PARAMETER;
    }
    
    return LOGISTIC_SUCCESS;
}

mlxlogistic_status_t mlxlogregcrossentropy(const mlxlogistic_model_t* model,
                                   const mat_t* X,
                                   const float* y,
                                   float* loss,
                                   mat_region_t* reg) {
    if (!model || !X || !y || !loss) {
        return LOGISTIC_NULL_POINTER;
    }
    
    size_t num_samples = X->row;
    
    // Get probabilities
    float* probs = model->workspace;
    
    mlxlogistic_status_t status = mlxlogregpredictproba(model, X, probs, reg);
    if (status != LOGISTIC_SUCCESS) {
        return status;
    }
    
    // Compute log loss
    float log_loss = 0.0f;
    
    if (model->num_classes <= 1) {
        // Binary classification
        for (size_t i = 0; i < num_samples; i++) {
            float p = probs[i];
            float t = y[i];
            
            // Clip probabilities for numerical stability
            if (p < 1e-15f) p = 1e-15f;
            if (p > 1.0f - 1e-15f) p = 1.0f - 1e-15f;
            
            log_loss -= t * mlxmatlogf(p) + (1.0f - t) * mlxmatlogf(1.0f - p);
        }
    }
    else {
        // Multi-class log loss
        // Placeholder for multi-class implementation
        return LOGISTIC_INVALID_PARAMETER;
    }
    
    // Average the loss
    *loss = log_loss / num_samples;
    
    // Add L2 regularization term if applicable
    if (model->config.l2_regularization > 0.0f) {
        float reg_sum = 0.0f;
        size_t start_idx = model->has_intercept ? 1 : 0; // Skip intercept
        
        for (size_t j = start_idx; j < model->weights.col; j++) {
            reg_sum += model->weights.data[j] * model->weights.data[j];
        }
        
        *loss += 0.5f * model->config.l2_regularization * reg_sum;
    }
    
    return LOGISTIC_SUCCESS;
}

const char* mlxlogregstrerror(mlxlogistic_status_t status) {
    switch (status) {
        case LOGISTIC_SUCCESS:
            return "Success";
        case LOGISTIC_NULL_POINTER:
            return "Null pointer";
        case LOGISTIC_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case LOGISTIC_NOT_CONVERGED:
            return "Did not converge within max iterations";
        case LOGISTIC_MEMORY_ERROR:
            return "Memory allocation error";
        case LOGISTIC_INVALID_PARAMETER:
            return "Invalid parameter";
        default:
            return "Unknown error";
    }
}
