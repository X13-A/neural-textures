#include "mlp_shared.h"
#include "cuda_check.h"
#include <iostream>
#include <vector>

__device__ inline float dot4(
    const float4& a,
    const float4& b)
{
    return a.x*b.x +
           a.y*b.y +
           a.z*b.z +
           a.w*b.w;
}

__device__ inline float4 relu(float4 v)
{
    v.x = fmaxf(v.x, 0.0f);
    v.y = fmaxf(v.y, 0.0f);
    v.z = fmaxf(v.z, 0.0f);
    v.w = fmaxf(v.w, 0.0f);
    return v;
}

void mlp_validate_config(MLP_Configuration* config)
{
    if (!config) return;
    for (int i = 0; i < config->num_layers; i++)
    {
        if (config->layer_sizes[i] % 4 != 0)
        {
            std::cerr << "Error: Layer size " << config->layer_sizes[i] << " is not a multiple of 4." << std::endl;
            exit(EXIT_FAILURE);
        }
        // TODO
    }
}

void mlp_load_host_parameters(float4* host_parameters, size_t host_parameters_count, MLP_Buffers* buffers)
{
    CUDA_CHECK(cudaMalloc(&buffers->d_parameters, host_parameters_count * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(buffers->d_parameters, host_parameters, host_parameters_count * sizeof(float4), cudaMemcpyHostToDevice));
}

void mlp_free(MLP_Buffers* buffers)
{
    if (!buffers) return;
    if (buffers->d_parameters)
    {
        CUDA_CHECK(cudaFree(buffers->d_parameters));
        buffers->d_parameters = nullptr;
    }
}

__device__ void eval_layer(
    const float4* parameters,
    size_t param_offset,
    const float4* previous,
    float4* current,
    uint32_t prev_quartets,
    uint32_t curr_quartets,
    ActivationFunction activation)
{
    for (uint32_t curr_q = 0; curr_q < curr_quartets; ++curr_q)
    {
        float4 neuron_value = {0,0,0,0};
    
        for (uint32_t prev_q = 0; prev_q < prev_quartets; ++prev_q)
        {
            const float4 prev_act = previous[prev_q];
    
            neuron_value.x += dot4(parameters[param_offset++], prev_act);
            neuron_value.y += dot4(parameters[param_offset++], prev_act);
            neuron_value.z += dot4(parameters[param_offset++], prev_act);
            neuron_value.w += dot4(parameters[param_offset++], prev_act);
        }
    
        const float4 bias = parameters[param_offset++];
        neuron_value.x += bias.x;
        neuron_value.y += bias.y;
        neuron_value.z += bias.z;
        neuron_value.w += bias.w;
    
        if (activation == ActivationFunction::ReLU)
        {
            neuron_value = relu(neuron_value);
        }
    
        current[curr_q] = neuron_value;
    }
}

__global__ void mlp_synthesis_kernel(int width, int height, Pixel* output_image, MLP_Configuration config, MLP_Buffers buffers)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    float u =        (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    float v = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    
    float4 input_quartets[1] = {};
    input_quartets[0].x = u;
    input_quartets[0].y = v;
    
    float4 activationsA[MAX_LAYER_QUARTETS];
    float4 activationsB[MAX_LAYER_QUARTETS];

    float4* ping = activationsA;
    float4* pong = activationsB;
    float4 final_output = {};

    for (int i = 0; i < config.num_layers - 1; i++)
    {
        float4* layer_input  = (i == 0) ? input_quartets : ping;
        float4* layer_output = (i == config.num_layers - 2) ? &final_output : pong;

        eval_layer(buffers.d_parameters,
            config.parameter_offsets[i],
            layer_input,
            layer_output,
            config.layer_sizes[i] / 4,
            config.layer_sizes[i + 1] / 4,
            config.activation_functions[i]);

        float4* tmp = ping; ping = pong; pong = tmp;
    }

    auto to_byte = [](float v)
    { 
        float clamped = fmaxf(0.0f, fminf(1.0f, v));
        return static_cast<unsigned char>(clamped * 255.0f);
    };

    Pixel p;
    p.r = to_byte(final_output.x);
    p.g = to_byte(final_output.y);
    p.b = to_byte(final_output.z);
    p.a = 255;
    
    output_image[y * width + x] = p;
}

void mlp_synthesis_run(int width, int height, Pixel* d_output_image, MLP_Configuration config, MLP_Buffers buffers)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    mlp_synthesis_kernel<<<gridSize, blockSize>>>(width, height, d_output_image, config, buffers);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}