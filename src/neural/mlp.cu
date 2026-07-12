#include "neural/mlp.h"
#include "neural/mlp_device.cuh"
#include "neural/utils/cuda_check.h"
#include <iostream>
#include <cstdlib>

// ============================================================================
// Lifecycle
// ============================================================================

void mlp_validate_config(const MLP_Configuration* config)
{
    if (!config) { return; }
    for (int i = 0; i < config->num_layers; i++)
    {
        if (config->layer_sizes[i] % 4 != 0)
        {
            std::cerr << "Layer size " << config->layer_sizes[i] << " is not a multiple of 4.\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

void mlp_allocate_parameters(const MLP_Configuration& config, MLP_Buffers* buffers)
{
    size_t total = 0;
    for (int i = 0; i < config.num_layers - 1; i++)
    {
        uint32_t prev_q = config.layer_sizes[i]     / 4;
        uint32_t curr_q = config.layer_sizes[i + 1] / 4;
        total += (size_t)curr_q * (prev_q * 4 + 1);
    }
    buffers->d_parameters_size = total * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&buffers->d_parameters, buffers->d_parameters_size));
}

void mlp_free(MLP_Buffers* buffers)
{
    if (!buffers) { return; }
    if (buffers->d_parameters)      { CUDA_CHECK(cudaFree(buffers->d_parameters));      buffers->d_parameters      = nullptr; }
    if (buffers->d_gradient_buffer) { CUDA_CHECK(cudaFree(buffers->d_gradient_buffer)); buffers->d_gradient_buffer = nullptr; }
    if (buffers->d_adam_mean)       { CUDA_CHECK(cudaFree(buffers->d_adam_mean));        buffers->d_adam_mean       = nullptr; }
    if (buffers->d_adam_variance)   { CUDA_CHECK(cudaFree(buffers->d_adam_variance));    buffers->d_adam_variance   = nullptr; }
}

// ============================================================================
// Kernels
// ============================================================================

__global__ void mlp_init_he_kernel(void* d_parameters_void, uint32_t total_params, MLP_Configuration config, uint32_t seed)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_params) return;

    int layer = 0;
    for (int i = 1; i < config.num_layers - 1; i++)
    {
        if ((uint32_t)config.parameter_offsets[i] <= idx) { layer = i; }
        else                                               { break; }
    }

    uint32_t prev_q   = config.layer_sizes[layer] / 4;
    uint32_t local    = idx - (uint32_t)config.parameter_offsets[layer];
    uint32_t block_sz = prev_q * 4 + 1;
    bool     is_bias  = (local % block_sz == prev_q * 4);

    float u_he = sqrtf(6.0f / (float)(prev_q * 4));
    uint32_t rng = idx * 1664525u + seed * 22695477u + 1013904223u;
    auto rand_weight = [&]() { return rand_float(rng) * 2.0f * u_he - u_he; };

    float4* d_parameters = (float4*)d_parameters_void;
    if (is_bias)
    {
        d_parameters[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    else
    {
        d_parameters[idx] = {rand_weight(), rand_weight(), rand_weight(), rand_weight()};
    }
}

__global__ void mlp_optimize_kernel(
    void*    d_parameters_void,
    int4*    gradient_buffer,
    float4*  d_adam_mean,
    float4*  d_adam_variance,
    uint32_t parameter_count,
    float    rcp_batch_size,
    float    learning_rate,
    uint32_t iteration,
    bool     use_sgd)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= parameter_count) return;

    float4 grad = unpack_float4(gradient_buffer[idx]);
    grad.x *= rcp_batch_size;
    grad.y *= rcp_batch_size;
    grad.z *= rcp_batch_size;
    grad.w *= rcp_batch_size;

    // step is the amount subtracted from each parameter
    float4 step;
    if (use_sgd)
    {
        // SGD
        step = {learning_rate * grad.x, learning_rate * grad.y,
                learning_rate * grad.z, learning_rate * grad.w};
    }
    else
    {
        // Adam
        float4 m = d_adam_mean[idx];
        float4 v = d_adam_variance[idx];

        m.x = kAdamBeta1 * m.x + (1.0f - kAdamBeta1) * grad.x;
        m.y = kAdamBeta1 * m.y + (1.0f - kAdamBeta1) * grad.y;
        m.z = kAdamBeta1 * m.z + (1.0f - kAdamBeta1) * grad.z;
        m.w = kAdamBeta1 * m.w + (1.0f - kAdamBeta1) * grad.w;

        v.x = kAdamBeta2 * v.x + (1.0f - kAdamBeta2) * grad.x * grad.x;
        v.y = kAdamBeta2 * v.y + (1.0f - kAdamBeta2) * grad.y * grad.y;
        v.z = kAdamBeta2 * v.z + (1.0f - kAdamBeta2) * grad.z * grad.z;
        v.w = kAdamBeta2 * v.w + (1.0f - kAdamBeta2) * grad.w * grad.w;

        d_adam_mean[idx]     = m;
        d_adam_variance[idx] = v;

        float bc1 = 1.0f - powf(kAdamBeta1, (float)iteration);
        float bc2 = 1.0f - powf(kAdamBeta2, (float)iteration);

        float4 m_hat = {m.x / bc1, m.y / bc1, m.z / bc1, m.w / bc1};
        float4 v_hat = {v.x / bc2, v.y / bc2, v.z / bc2, v.w / bc2};

        step = {learning_rate * m_hat.x / (sqrtf(v_hat.x) + kAdamEpsilon),
                learning_rate * m_hat.y / (sqrtf(v_hat.y) + kAdamEpsilon),
                learning_rate * m_hat.z / (sqrtf(v_hat.z) + kAdamEpsilon),
                learning_rate * m_hat.w / (sqrtf(v_hat.w) + kAdamEpsilon)};
    }

    float4* d_parameters = (float4*)d_parameters_void;
    float4 param = d_parameters[idx];
    param.x -= step.x;
    param.y -= step.y;
    param.z -= step.z;
    param.w -= step.w;
    d_parameters[idx] = param;

    gradient_buffer[idx] = {0, 0, 0, 0};
}

// ============================================================================
// Host
// ============================================================================

void mlp_allocate_training_buffers(MLP_Buffers* buffers)
{
    size_t count = buffers->d_parameters_size / sizeof(float4);

    CUDA_CHECK(cudaMalloc(&buffers->d_gradient_buffer, count * sizeof(int4)));
    CUDA_CHECK(cudaMemset(buffers->d_gradient_buffer, 0, count * sizeof(int4)));

    CUDA_CHECK(cudaMalloc(&buffers->d_adam_mean,     count * sizeof(float4)));
    CUDA_CHECK(cudaMemset(buffers->d_adam_mean,     0, count * sizeof(float4)));

    CUDA_CHECK(cudaMalloc(&buffers->d_adam_variance, count * sizeof(float4)));
    CUDA_CHECK(cudaMemset(buffers->d_adam_variance, 0, count * sizeof(float4)));
}

void mlp_init_he_weights(const MLP_Configuration& config, MLP_Buffers* buffers, uint32_t seed)
{
    uint32_t total = (uint32_t)(buffers->d_parameters_size / sizeof(float4));
    constexpr int kBlockSize = 256;
    int grid = (total + kBlockSize - 1) / kBlockSize;
    mlp_init_he_kernel<<<grid, kBlockSize>>>(buffers->d_parameters, total, config, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void mlp_reset_training(const MLP_Configuration& config, MLP_Buffers* buffers, uint32_t seed)
{
    mlp_init_he_weights(config, buffers, seed);

    size_t count = buffers->d_parameters_size / sizeof(float4);

    CUDA_CHECK(cudaMemset(buffers->d_gradient_buffer, 0, count * sizeof(int4)));
    CUDA_CHECK(cudaMemset(buffers->d_adam_mean, 0, count * sizeof(float4)));
    CUDA_CHECK(cudaMemset(buffers->d_adam_variance, 0, count * sizeof(float4)));
}

void mlp_clear_adam(MLP_Buffers* buffers)
{
    size_t count = buffers->d_parameters_size / sizeof(float4);

    CUDA_CHECK(cudaMemset(buffers->d_adam_mean,     0, count * sizeof(float4)));
    CUDA_CHECK(cudaMemset(buffers->d_adam_variance, 0, count * sizeof(float4)));
}

void mlp_optimize(MLP_Buffers* buffers, float learning_rate, float rcp_batch_size,
                  uint32_t iteration, bool use_sgd)
{
    uint32_t param_count = (uint32_t)(buffers->d_parameters_size / sizeof(float4));
    constexpr int kBlockSize = 256;
    int grid = (param_count + kBlockSize - 1) / kBlockSize;

    mlp_optimize_kernel<<<grid, kBlockSize>>>(
        buffers->d_parameters, buffers->d_gradient_buffer,
        buffers->d_adam_mean, buffers->d_adam_variance,
        param_count, rcp_batch_size, learning_rate,
        iteration + 1, // 1-based for Adam bias correction
        use_sgd);
    CUDA_CHECK(cudaGetLastError());
}
