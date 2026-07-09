#pragma once
#include "neural/mlp_types.h"

void mlp_validate_config(const MLP_Configuration* config);
void mlp_allocate_parameters(const MLP_Configuration& config, MLP_Buffers* buffers);
void mlp_free(MLP_Buffers* buffers);

// Allocates gradient + Adam buffers
void mlp_allocate_training_buffers(MLP_Buffers* buffers);

// Initialises weights with He uniform
void mlp_init_he_weights(const MLP_Configuration& config, MLP_Buffers* buffers, uint32_t seed = 0);

// Resets all parameters and optimizer
void mlp_reset_training(const MLP_Configuration& config, MLP_Buffers* buffers, uint32_t seed = 0);

// Clears only Adam first/second moment buffers
void mlp_clear_adam(MLP_Buffers* buffers);

// Runs one optimizer step over all parameters and clears the gradient buffer.
// Assumes gradients have already been accumulated.
void mlp_optimize(MLP_Buffers* buffers, float learning_rate, float rcp_batch_size,
                  uint32_t iteration, bool use_sgd = false);
