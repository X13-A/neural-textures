#include "tasks/image_fit.h"
#include "neural/mlp.h"
#include "neural/mlp_device.cuh"
#include "neural/hashgrid.h"
#include "neural/hashgrid_device.cuh"
#include "neural/utils/cuda_check.h"

// ============================================================================
// Architecture
// ============================================================================

MLP_Configuration image_fit_build_mlp_config()
{
    MLP_Configuration config = {};

    config.layer_sizes[0] = HG_INPUT_SIZE;
    config.layer_sizes[1] = 32;
    config.layer_sizes[2] = 32;
    config.layer_sizes[3] = 4;
    config.num_layers = 4;

    config.activation_functions[0] = ActivationFunction::ReLU;
    config.activation_functions[1] = ActivationFunction::ReLU;
    config.activation_functions[2] = ActivationFunction::None;

    size_t   offset     = 0;
    uint32_t act_offset = 0;
    for (int i = 0; i < config.num_layers - 1; i++)
    {
        config.parameter_offsets[i]  = offset;
        config.activation_offsets[i] = act_offset;

        uint32_t prev_q = config.layer_sizes[i]     / 4;
        uint32_t curr_q = config.layer_sizes[i + 1] / 4;
        offset     += curr_q * (prev_q * 4 + 1);
        act_offset += curr_q;
    }

    return config;
}

// ============================================================================
// Inference
// ============================================================================

__global__ void image_fit_synthesis_kernel(
    int width, int height, uchar4* output,
    MLP_Configuration config, MLP_Buffers buffers,
    HashGrid_Configuration hg_config, const float4* hg_features,
    bool use_hashgrid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    float u =        (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    float v = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

    float4 encoded[HG_INPUT_QUARTETS];
    if (use_hashgrid)
    {
        hg_encode(hg_config, hg_features, {u, v}, encoded);
    }
    else
    {
        encoded[0] = {u, v, 0.0f, 0.0f};
    }

    float4 out = {0.0f, 0.0f, 0.0f, 0.0f};
    forward_inference(
        encoded, &out,
        (const float4*)buffers.d_parameters, &config);

    auto to_byte = [](float f)
    {
        return static_cast<unsigned char>(fmaxf(0.0f, fminf(1.0f, f)) * 255.0f);
    };

    output[y * width + x] = { to_byte(out.x), to_byte(out.y), to_byte(out.z), 255 };
}

void image_fit_synthesis_run(
    int width, int height, uchar4* d_output,
    MLP_Configuration config, MLP_Buffers buffers,
    HashGrid_Configuration hg_config, const HashGrid_Buffers& hg_buffers,
    bool use_hashgrid)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    image_fit_synthesis_kernel<<<grid, block>>>(
        width, height, d_output,
        config, buffers,
        hg_config, hg_buffers.d_features,
        use_hashgrid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Training
// ============================================================================

__global__ void image_fit_train_kernel(
    const uchar4* target_image,
    int image_width,
    int image_height,
    const float4* d_parameters,
    float4*       gradient_buffer,
    MLP_Configuration config,
    uint32_t      iteration,
    uint32_t      rng_seed,
    float*        d_loss_output,
    bool          use_hashgrid,
    HashGrid_Configuration hg_config,
    const float4* hg_features,
    float4*       hg_gradient)
{
    // Draw random UV and load target color
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t rng = thread_id * 1664525u + (iteration + rng_seed) * 22695477u + 1013904223u;
    float2 input = {rand_float(rng), rand_float(rng)};

    int tx = min((int)(input.x * image_width),  image_width  - 1);
    int ty = min((int)(input.y * image_height), image_height - 1);
    uchar4 target_texel = target_image[ty * image_width + tx];
    float3 target_color = {target_texel.x / 255.0f, target_texel.y / 255.0f, target_texel.z / 255.0f};

    // Encode input UV using hash-grid
    float4 encoded[HG_INPUT_QUARTETS];
    if (use_hashgrid)
        hg_encode(hg_config, hg_features, input, encoded);
    else
        encoded[0] = {input.x, input.y, 0.0f, 0.0f};

    // Perform forward pass
    float4 activations[ACTIVATION_QUARTETS_PER_NETWORK];
    forward_training(encoded, activations, (const float4*)d_parameters, &config);

    // Locate output activations
    const float4* out = activations + config.activation_offsets[config.num_layers - 2];

    // L2 loss gradient dL/d(output) = 2 * (output - target)
    // The factor of 2 is not necessary here as it just scales the learning rate
    float4 output_grad =
    {
        out[0].x - target_color.x,
        out[0].y - target_color.y,
        out[0].z - target_color.z,
        // The 4th channel's gradient is zeroed-out to leave it untrained
        0.0f
    };

    // Perform backwards pass
    float4 input_delta[HG_INPUT_QUARTETS];
    backprop_training(encoded, input_delta, &output_grad, activations, (const float4*)d_parameters, gradient_buffer, &config);

    if (use_hashgrid)
    {
        for (uint32_t level = 0; level < HG_NUM_LEVELS; level++)
        {
            hg_accumulate_gradient(hg_config, hg_gradient, input, input_delta, level);
        }
    }

    float3 diff = {out[0].x - target_color.x, out[0].y - target_color.y, out[0].z - target_color.z};
    atomicAdd(d_loss_output, (diff.x*diff.x + diff.y*diff.y + diff.z*diff.z) / 3.0f);
}

float image_fit_train_step(
    const uchar4* d_target_image,
    int image_width,
    int image_height,
    int batch_size,
    float learning_rate,
    uint32_t iteration,
    MLP_Configuration config,
    MLP_Buffers buffers,
    HashGrid_Configuration hg_config,
    HashGrid_Buffers hg_buffers,
    bool use_hashgrid,
    KernelTimings* out_timings,
    uint32_t rng_seed,
    bool use_sgd)
{
    constexpr int kBlockSize = 256;
    int backprop_grid = (batch_size + kBlockSize - 1) / kBlockSize;
    float rcp_batch   = 1.0f / (float)batch_size;

    float* d_loss = nullptr;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    if (out_timings)
    {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }

    // Accumulate gradients from the batch
    if (out_timings) { CUDA_CHECK(cudaEventRecord(ev_start)); }
    image_fit_train_kernel<<<backprop_grid, kBlockSize>>>(
        d_target_image, image_width, image_height,
        buffers.d_parameters, buffers.d_gradient_buffer,
        config, iteration, rng_seed, d_loss,
        use_hashgrid, hg_config, hg_buffers.d_features, hg_buffers.d_gradient);
    CUDA_CHECK(cudaGetLastError());
    if (out_timings)
    {
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&out_timings->backprop_ms, ev_start, ev_stop));
    }

    // Optimize the MLP parameters
    if (out_timings) { CUDA_CHECK(cudaEventRecord(ev_start)); }
    mlp_optimize(&buffers, learning_rate, rcp_batch, iteration, use_sgd);
    if (out_timings)
    {
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&out_timings->optimize_ms, ev_start, ev_stop));
    }

    // Optimize the hash-grid features
    if (use_hashgrid && out_timings)
    {
        CUDA_CHECK(cudaEventRecord(ev_start));
        hg_optimize(hg_config, &hg_buffers, learning_rate, rcp_batch, iteration, use_sgd);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&out_timings->hg_gradient_ms, ev_start, ev_stop));
    }
    else if (use_hashgrid)
    {
        hg_optimize(hg_config, &hg_buffers, learning_rate, rcp_batch, iteration, use_sgd);
    }

    if (out_timings)
    {
        out_timings->total_ms = out_timings->backprop_ms + out_timings->optimize_ms + out_timings->hg_gradient_ms;
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float host_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));

    return host_loss / (float)batch_size;
}
