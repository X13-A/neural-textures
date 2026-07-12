#include "neural/hashgrid_device.cuh"
#include "neural/utils/cuda_check.h"
#include <iostream>
#include <algorithm>

HashGrid_Configuration hg_build_config()
{
    HashGrid_Configuration config = {};
    uint32_t offset = 0;

    for (uint32_t level = 0; level < HG_NUM_LEVELS; level++)
    {
        uint32_t res        = HG_BASE_RESOLUTION << level;
        uint32_t table_size = std::min(res * res, HG_HASH_THRESHOLD);

        config.resolutions[level]   = res;
        config.table_sizes[level]   = table_size;
        config.entry_offsets[level] = offset;

        offset += table_size * HG_FEATURE_QUARTETS;
    }

    config.total_float4s = offset;

    std::cout << "Hash grid: " << HG_NUM_LEVELS << " levels, "
              << config.total_float4s << " float4 entries total" << std::endl;

    return config;
}

__global__ void hg_init_kernel(float4* features, uint32_t total_float4s, uint32_t seed)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_float4s) { return; }

    uint32_t rng = idx * 1664525u + seed * 22695477u + 1013904223u;
    // Small uniform range keeps features near zero at init
    auto r = [&]() { return rand_float(rng) * 2e-4f - 1e-4f; };
    features[idx] = {r(), r(), r(), r()};
}

__global__ void hg_optimize_kernel(
    float4*  features,
    float4*  gradient,
    float4*  adam_mean,
    float4*  adam_variance,
    uint32_t total_float4s,
    float    rcp_batch_size,
    float    learning_rate,
    uint32_t iteration,
    bool     use_sgd)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_float4s) { return; }

    float4 grad = gradient[idx];
    grad.x *= rcp_batch_size;
    grad.y *= rcp_batch_size;
    grad.z *= rcp_batch_size;
    grad.w *= rcp_batch_size;

    // step is the amount subtracted from each feature value.
    float4 step;
    if (use_sgd)
    {
        // SGD
        step = {learning_rate * grad.x, learning_rate * grad.y,
                learning_rate * grad.z, learning_rate * grad.w};
    }
    else
    {
        // ADAM
        float4 m = adam_mean[idx];
        float4 v = adam_variance[idx];

        m.x = kAdamBeta1 * m.x + (1.0f - kAdamBeta1) * grad.x;
        m.y = kAdamBeta1 * m.y + (1.0f - kAdamBeta1) * grad.y;
        m.z = kAdamBeta1 * m.z + (1.0f - kAdamBeta1) * grad.z;
        m.w = kAdamBeta1 * m.w + (1.0f - kAdamBeta1) * grad.w;

        v.x = kAdamBeta2 * v.x + (1.0f - kAdamBeta2) * grad.x * grad.x;
        v.y = kAdamBeta2 * v.y + (1.0f - kAdamBeta2) * grad.y * grad.y;
        v.z = kAdamBeta2 * v.z + (1.0f - kAdamBeta2) * grad.z * grad.z;
        v.w = kAdamBeta2 * v.w + (1.0f - kAdamBeta2) * grad.w * grad.w;

        adam_mean[idx]     = m;
        adam_variance[idx] = v;

        float bc1 = 1.0f - powf(kAdamBeta1, (float)iteration);
        float bc2 = 1.0f - powf(kAdamBeta2, (float)iteration);

        float4 m_hat = {m.x / bc1, m.y / bc1, m.z / bc1, m.w / bc1};
        float4 v_hat = {v.x / bc2, v.y / bc2, v.z / bc2, v.w / bc2};

        step = {learning_rate * m_hat.x / (sqrtf(v_hat.x) + kAdamEpsilon),
                learning_rate * m_hat.y / (sqrtf(v_hat.y) + kAdamEpsilon),
                learning_rate * m_hat.z / (sqrtf(v_hat.z) + kAdamEpsilon),
                learning_rate * m_hat.w / (sqrtf(v_hat.w) + kAdamEpsilon)};
    }

    float4 feat = features[idx];
    feat.x -= step.x;
    feat.y -= step.y;
    feat.z -= step.z;
    feat.w -= step.w;
    features[idx] = feat;

    gradient[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
}

void hg_allocate(const HashGrid_Configuration& config, HashGrid_Buffers* buffers)
{
    size_t bytes = config.total_float4s * sizeof(float4);

    CUDA_CHECK(cudaMalloc(&buffers->d_features,      bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_gradient,      bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_adam_mean,     bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_adam_variance, bytes));

    CUDA_CHECK(cudaMemset(buffers->d_gradient,      0, bytes));
    CUDA_CHECK(cudaMemset(buffers->d_adam_mean,     0, bytes));
    CUDA_CHECK(cudaMemset(buffers->d_adam_variance, 0, bytes));
}

void hg_init_features(const HashGrid_Configuration& config, HashGrid_Buffers* buffers, uint32_t seed)
{
    constexpr int kBlockSize = 256;
    int grid = (config.total_float4s + kBlockSize - 1) / kBlockSize;
    hg_init_kernel<<<grid, kBlockSize>>>(buffers->d_features, config.total_float4s, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void hg_optimize(const HashGrid_Configuration& config, HashGrid_Buffers* buffers,
                 float learning_rate, float rcp_batch_size, uint32_t iteration,
                 bool use_sgd)
{
    constexpr int kBlockSize = 256;
    int grid = (config.total_float4s + kBlockSize - 1) / kBlockSize;
    hg_optimize_kernel<<<grid, kBlockSize>>>(
        buffers->d_features, buffers->d_gradient,
        buffers->d_adam_mean, buffers->d_adam_variance,
        config.total_float4s,
        rcp_batch_size,
        learning_rate,
        iteration + 1, // 1-based for Adam bias correction
        use_sgd);
    CUDA_CHECK(cudaGetLastError());
}

void hg_reset_training(const HashGrid_Configuration& config, HashGrid_Buffers* buffers, uint32_t seed)
{
    hg_init_features(config, buffers, seed);
    size_t bytes = config.total_float4s * sizeof(float4);
    CUDA_CHECK(cudaMemset(buffers->d_gradient, 0, bytes));
    CUDA_CHECK(cudaMemset(buffers->d_adam_mean, 0, bytes));
    CUDA_CHECK(cudaMemset(buffers->d_adam_variance, 0, bytes));
}

void hg_clear_adam(const HashGrid_Configuration& config, HashGrid_Buffers* buffers)
{
    size_t bytes_f4 = config.total_float4s * sizeof(float4);
    CUDA_CHECK(cudaMemset(buffers->d_adam_mean,     0, bytes_f4));
    CUDA_CHECK(cudaMemset(buffers->d_adam_variance, 0, bytes_f4));
}

void hg_free(HashGrid_Buffers* buffers)
{
    if (!buffers) { return; }
    if (buffers->d_features)      { CUDA_CHECK(cudaFree(buffers->d_features));      buffers->d_features      = nullptr; }
    if (buffers->d_gradient)      { CUDA_CHECK(cudaFree(buffers->d_gradient));      buffers->d_gradient      = nullptr; }
    if (buffers->d_adam_mean)     { CUDA_CHECK(cudaFree(buffers->d_adam_mean));     buffers->d_adam_mean     = nullptr; }
    if (buffers->d_adam_variance) { CUDA_CHECK(cudaFree(buffers->d_adam_variance)); buffers->d_adam_variance = nullptr; }
}

// Picks scalar component c (0..3) out of a float4.
__device__ inline float hg_vis_component(float4 v, int c)
{
    return (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
}

__global__ void hg_visualize_kernel(
    int width, int height, uchar4* output,
    HashGrid_Configuration config, const float4* features,
    int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    float u =        (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    float v = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

    float4 encoded[HG_INPUT_QUARTETS];
    hg_encode(config, features, {u, v}, encoded);

    float value = hg_vis_component(encoded[channel / 4], channel % 4);
    unsigned char grey = static_cast<unsigned char>(fmaxf(0.0f, fminf(1.0f, value)) * 255.0f);

    output[y * width + x] = { grey, grey, grey, 255 };
}

void hg_visualize_run(int width, int height, uchar4* d_output,
                      const HashGrid_Configuration& config, const HashGrid_Buffers& buffers,
                      int channel)
{
    channel = std::max(0, std::min(channel, (int)HG_INPUT_SIZE - 1));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    hg_visualize_kernel<<<grid, block>>>(width, height, d_output, config, buffers.d_features, channel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
