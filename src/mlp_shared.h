#pragma once
#include <cuda_runtime.h>

#define MAX_LAYER_QUARTETS 64
#define MAX_LAYERS 16

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
    int num_layers;
};

struct MLP_Buffers
{
    float4* d_parameters = nullptr;
    size_t d_parameters_size = 0;
};

struct Pixel { unsigned char r, g, b, a; };