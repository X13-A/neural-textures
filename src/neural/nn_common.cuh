#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Shared device code used by both the MLP and the hash grid

static constexpr float kAdamBeta1   = 0.9f;
static constexpr float kAdamBeta2   = 0.999f;
static constexpr float kAdamEpsilon = 1e-8f;

// RNG

__device__ inline uint32_t lcg_step(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ inline float rand_float(uint32_t& state)
{
    return (float)(lcg_step(state) >> 8) * (1.0f / 16777216.0f);
}

// Gradient accumulation

__device__ inline void accumulate_gradient(float4* gradient_buffer, uint32_t idx, float4 gradient)
{
    atomicAdd(&gradient_buffer[idx].x, gradient.x);
    atomicAdd(&gradient_buffer[idx].y, gradient.y);
    atomicAdd(&gradient_buffer[idx].z, gradient.z);
    atomicAdd(&gradient_buffer[idx].w, gradient.w);
}
