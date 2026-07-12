#pragma once
#include "neural/mlp_types.h"
#include "neural/nn_common.cuh"

__device__ inline float dot4(const float4& a, const float4& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device__ inline float4 relu(float4 v)
{
    v.x = fmaxf(v.x, 0.0f);
    v.y = fmaxf(v.y, 0.0f);
    v.z = fmaxf(v.z, 0.0f);
    v.w = fmaxf(v.w, 0.0f);
    return v;
}

__device__ inline void eval_layer(
    const float4* parameters,
    size_t param_offset,
    const float4* previous,
    float4* current,
    uint32_t prev_quartets,
    uint32_t curr_quartets,
    ActivationFunction activation)
{
    // Iterate over current layer's neurons in groups of 4 (quartets)
    for (uint32_t curr_q = 0; curr_q < curr_quartets; ++curr_q)
    {
        float neuron_value_x = 0.0f, neuron_value_y = 0.0f, neuron_value_z = 0.0f, neuron_value_w = 0.0f;

        // Apply weights
        for (uint32_t prev_q = 0; prev_q < prev_quartets; ++prev_q)
        {
            const float4 prev_act = previous[prev_q];
            const float4 w0 = parameters[param_offset++];
            const float4 w1 = parameters[param_offset++];
            const float4 w2 = parameters[param_offset++];
            const float4 w3 = parameters[param_offset++];

            neuron_value_x += dot4(prev_act, w0);
            neuron_value_y += dot4(prev_act, w1);
            neuron_value_z += dot4(prev_act, w2);
            neuron_value_w += dot4(prev_act, w3);
        }

        // Apply bias
        const float4 bias = parameters[param_offset++];
        neuron_value_x += bias.x;
        neuron_value_y += bias.y;
        neuron_value_z += bias.z;
        neuron_value_w += bias.w;

        // Apply activation function
        float4 neuron_value;
        if (activation == ActivationFunction::ReLU)
        {
            neuron_value = {
                fmaxf(neuron_value_x, 0.0f),
                fmaxf(neuron_value_y, 0.0f),
                fmaxf(neuron_value_z, 0.0f),
                fmaxf(neuron_value_w, 0.0f)
            };
        }
        else
        {
            neuron_value = {neuron_value_x, neuron_value_y, neuron_value_z, neuron_value_w};
        }

        current[curr_q] = neuron_value;
    }
}

// ============================================================================
// Forward
// ============================================================================

__device__ inline void forward_inference(
    const float4* input_features,
    float4*       output,
    const float4* d_parameters,
    const MLP_Configuration* config)
{
    float4 activationsA[MAX_LAYER_QUARTETS];
    float4 activationsB[MAX_LAYER_QUARTETS];
    float4* ping = activationsA;
    float4* pong = activationsB;

    for (int i = 0; i < config->num_layers - 1; i++)
    {
        const float4* layer_input  = (i == 0)                      ? input_features : ping;
        float4*       layer_output = (i == config->num_layers - 2) ? output         : pong;

        eval_layer(
            d_parameters,
            config->parameter_offsets[i],
            layer_input, layer_output,
            config->layer_sizes[i]     / 4,
            config->layer_sizes[i + 1] / 4,
            config->activation_functions[i]);

        float4* tmp = ping; ping = pong; pong = tmp;
    }
}

// Same but stores all activations for backpropagation
__device__ inline void forward_training(
    const float4* input_features,
    float4*       activations,
    const float4* d_parameters,
    const MLP_Configuration* config)
{
    for (int i = 0; i < config->num_layers - 1; i++)
    {
        const float4* layer_input = (i == 0)
            ? input_features
            : activations + config->activation_offsets[i - 1];

        eval_layer(d_parameters,
            config->parameter_offsets[i],
            layer_input,
            activations + config->activation_offsets[i],
            config->layer_sizes[i]     / 4,
            config->layer_sizes[i + 1] / 4,
            config->activation_functions[i]);
    }
}

// ============================================================================
// Backward pass
// ============================================================================

__device__ inline void backprop_training(
    const float4*            input_features,
    float4*                  input_delta,
    const float4*            output_grad,
    const float4*            activations,
    const float4*            d_parameters,
    float4*                  gradient_buffer,
    const MLP_Configuration* config)
{
    int last = config->num_layers - 2;

    // Per-sample delta buffers for the current and previous layers
    float4 delta_curr[MAX_LAYER_QUARTETS];
    float4 delta_prev[MAX_LAYER_QUARTETS];

    // Seed output layer with supplied gradient
    uint32_t output_quartets = config->layer_sizes[config->num_layers - 1] / 4;
    const float4* output_act = activations + config->activation_offsets[last];
    for (uint32_t q = 0; q < output_quartets; q++)
    {
        delta_curr[q] = output_grad[q];
    }

    // Apply ReLU derivative to the output layer
    if (config->activation_functions[last] == ActivationFunction::ReLU)
    {
        for (uint32_t q = 0; q < output_quartets; q++)
        {
            delta_curr[q].x *= (output_act[q].x > 0.0f) ? 1.0f : 0.0f;
            delta_curr[q].y *= (output_act[q].y > 0.0f) ? 1.0f : 0.0f;
            delta_curr[q].z *= (output_act[q].z > 0.0f) ? 1.0f : 0.0f;
            delta_curr[q].w *= (output_act[q].w > 0.0f) ? 1.0f : 0.0f;
        }
    }

    // Start backpropagation from the last layer down to the first hidden layer
    for (int i = last; i >= 0; i--)
    {
        uint32_t curr_quartets = config->layer_sizes[i + 1] / 4;
        uint32_t prev_quartets = config->layer_sizes[i] / 4;
        size_t   layer_param_base = config->parameter_offsets[i];

        const float4* prev_act = (i == 0) ? input_features : activations + config->activation_offsets[i - 1];

        for (uint32_t q = 0; q < prev_quartets; q++)
        {
            delta_prev[q] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        for (uint32_t curr_q = 0; curr_q < curr_quartets; curr_q++)
        {
            float4 d = delta_curr[curr_q];
            size_t q_base = layer_param_base + curr_q * (prev_quartets * 4 + 1);

            for (uint32_t prev_q = 0; prev_q < prev_quartets; prev_q++)
            {
                size_t w_base = q_base + prev_q * 4;
                float4 pa = prev_act[prev_q];

                accumulate_gradient(gradient_buffer, w_base + 0, {d.x*pa.x, d.x*pa.y, d.x*pa.z, d.x*pa.w});
                accumulate_gradient(gradient_buffer, w_base + 1, {d.y*pa.x, d.y*pa.y, d.y*pa.z, d.y*pa.w});
                accumulate_gradient(gradient_buffer, w_base + 2, {d.z*pa.x, d.z*pa.y, d.z*pa.z, d.z*pa.w});
                accumulate_gradient(gradient_buffer, w_base + 3, {d.w*pa.x, d.w*pa.y, d.w*pa.z, d.w*pa.w});

                const float4 w0 = d_parameters[w_base + 0];
                const float4 w1 = d_parameters[w_base + 1];
                const float4 w2 = d_parameters[w_base + 2];
                const float4 w3 = d_parameters[w_base + 3];

                delta_prev[prev_q].x += d.x*w0.x + d.y*w1.x + d.z*w2.x + d.w*w3.x;
                delta_prev[prev_q].y += d.x*w0.y + d.y*w1.y + d.z*w2.y + d.w*w3.y;
                delta_prev[prev_q].z += d.x*w0.z + d.y*w1.z + d.z*w2.z + d.w*w3.z;
                delta_prev[prev_q].w += d.x*w0.w + d.y*w1.w + d.z*w2.w + d.w*w3.w;
            }

            accumulate_gradient(gradient_buffer, q_base + prev_quartets * 4, d);
        }

        // Apply ReLU derivative (skipping input layer)
        if (i > 0 && config->activation_functions[i - 1] == ActivationFunction::ReLU)
        {
            const float4* layer_i_act = activations + config->activation_offsets[i - 1];
            for (uint32_t q = 0; q < prev_quartets; q++)
            {
                delta_prev[q].x *= (layer_i_act[q].x > 0.0f) ? 1.0f : 0.0f;
                delta_prev[q].y *= (layer_i_act[q].y > 0.0f) ? 1.0f : 0.0f;
                delta_prev[q].z *= (layer_i_act[q].z > 0.0f) ? 1.0f : 0.0f;
                delta_prev[q].w *= (layer_i_act[q].w > 0.0f) ? 1.0f : 0.0f;
            }
        }

        for (uint32_t q = 0; q < prev_quartets; q++)
        {
            delta_curr[q] = delta_prev[q];
        }
    }

    uint32_t input_quartets = config->layer_sizes[0] / 4;
    for (uint32_t q = 0; q < input_quartets; q++)
    {
        input_delta[q] = delta_curr[q];
    }
}
