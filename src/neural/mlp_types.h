#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Should be defined by the user in the application layer
// Fit this to your neural network size
#ifndef MAX_LAYER_QUARTETS
    #define MAX_LAYER_QUARTETS 16
#endif

#ifndef MAX_LAYERS
    #define MAX_LAYERS 8
#endif

#define ACTIVATION_QUARTETS_PER_NETWORK (MAX_LAYER_QUARTETS * MAX_LAYERS)

enum class ActivationFunction
{
    ReLU,
    None
};

struct MLP_Configuration
{
    uint32_t layer_sizes[MAX_LAYERS];
    ActivationFunction activation_functions[MAX_LAYERS];
    size_t parameter_offsets[MAX_LAYERS];
    uint32_t activation_offsets[MAX_LAYERS];
    int num_layers;
};

struct MLP_Buffers
{
    float4* d_parameters      = nullptr;
    size_t  d_parameters_size = 0;
    int4*   d_gradient_buffer = nullptr;
    float4* d_adam_mean       = nullptr;
    float4* d_adam_variance   = nullptr;
};
