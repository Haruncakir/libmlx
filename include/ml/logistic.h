#ifndef MLX_LOGISTIC_H
#define MLX_LOGISTIC_H

#include "../matrix/matrix.h"
#include "../matrix/matpool.h"

typedef unsigned char mlxlogistic_status_t;
typedef enum mlxbool {
    mlxboolfalse,
    mlxbooltrue
} mlxbool;

#define LOGISTIC_SUCCESS ((mlxlogistic_status_t)0)
#define LOGISTIC_NULL_POINTER ((mlxlogistic_status_t)1)
#define LOGISTIC_DIMENSION_MISMATCH ((mlxlogistic_status_t)2)
#define LOGISTIC_NOT_CONVERGED ((mlxlogistic_status_t)3)
#define LOGISTIC_MEMORY_ERROR ((mlxlogistic_status_t)4)
#define LOGISTIC_INVALID_PARAMETER ((mlxlogistic_status_t)5)

typedef struct {
    float learning_rate;      /**< Learning rate for gradient descent */
    float l2_regularization;  /**< L2 regularization strength */
    size_t max_iterations;    /**< Maximum number of training iterations */
    float convergence_tol;    /**< Convergence tolerance */
    mlxbool fit_intercept;       /**< Whether to fit an intercept term */
    mlxbool verbose;     //???   /**< Whether to print training progress */
} mlxlogistic_config_t;

/**
 * @brief Logistic regression model structure
 * 
 * This structure holds the model parameters and metadata for
 * a logistic regression model. The weights are stored in a matrix
 * structure, which can be allocated from a memory pool.
 */
typedef struct {
    mat_t weights;            /**< Model weights (including intercept if used) */
    mlxbool has_intercept;       /**< Whether model includes an intercept term */
    size_t num_features;      /**< Number of features (excluding intercept) */
    size_t num_classes;       /**< Number of classes (1 for binary) */
    float* workspace;         /**< Temporary computation workspace */
    size_t workspace_size;    /**< Size of workspace in floats */
    mlxlogistic_config_t config; /**< Model configuration parameters */
} mlxlogistic_model_t;

/**
 * @brief Initialize a default logistic regression configuration
 * 
 * @param config Pointer to configuration structure to initialize
 * @return logistic_status_t Status code
 */
mlxlogistic_status_t mlxlogregconfiginit(mlxlogistic_config_t *config);

/**
 * @brief Initialize a logistic regression model
 * 
 * @param model Pointer to model structure to initialize
 * @param reg Memory region for allocating model parameters
 * @param num_features Number of input features
 * @param num_classes Number of output classes (1 for binary classification)
 * @param config Model configuration (NULL for defaults)
 * @return logistic_status_t Status code
 */
mlxlogistic_status_t mlxlogreginit(mlxlogistic_model_t* model, 
                               mat_region_t* reg,
                               size_t num_features, 
                               size_t num_classes,
                               const mlxlogistic_config_t* config);

/**
 * @brief Train a logistic regression model using gradient descent
 * 
 * @param model Pointer to initialized model
 * @param X Feature matrix (rows=samples, cols=features)
 * @param y Target vector (binary labels 0/1 for each sample)
 * @param reg Memory region for temporary allocations
 * @return logistic_status_t Status code
 */
mlxlogistic_status_t mlxlogregtrain(mlxlogistic_model_t* model,
                                const mat_t* X,
                                const float* y,
                                mat_region_t* reg);

/**
 * @brief Predict probabilities using a logistic regression model
 * 
 * @param model Trained logistic regression model
 * @param X Feature matrix (rows=samples, cols=features)
 * @param probs Output probability predictions (must be pre-allocated)
 * @param reg Memory region for temporary allocations
 * @return logistic_status_t Status code
 */
mlxlogistic_status_t mlxlogregpredictproba(const mlxlogistic_model_t* model,
                                        const mat_t* X,
                                        float* probs,
                                        mat_region_t* reg);

/**
 * @brief Predict class labels using a logistic regression model
 * 
 * @param model Trained logistic regression model
 * @param X Feature matrix (rows=samples, cols=features)
 * @param labels Output class predictions (must be pre-allocated)
 * @param reg Memory region for temporary allocations
 * @return logistic_status_t Status code
 */
mlxlogistic_status_t mlxlogregpredict(const mlxlogistic_model_t* model,
                                  const mat_t* X,
                                  float* labels,
                                  float threshold,
                                  mat_region_t* reg);

/**
 * @brief Calculate the log loss (cross-entropy) for model evaluation
 * 
 * @param model Trained logistic regression model
 * @param X Feature matrix (rows=samples, cols=features)
 * @param y True target labels
 * @param loss Pointer to store the calculated loss
 * @param reg Memory region for temporary allocations
 * @return logistic_status_t Status code
 */
mlxlogistic_status_t mlxlogregcrossentropy(const mlxlogistic_model_t* model,
                                   const mat_t* X,
                                   const float* y,
                                   float* loss,
                                   mat_region_t* reg);

/**
 * @brief Get a string description of a status code
 * 
 * @param status Status code
 * @return const char* String description
 */
const char* mlxlogregstrerror(mlxlogistic_status_t status);

#endif // MLX_LOGISTIC_H
