#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Shared device code used by both the MLP and the hash grid

static constexpr float FLOAT_PACKING_CONSTANT = 65536.0f;

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

// Gradient packing

__device__ inline int4 pack_float4(float4 x)
{
    return {(int)(x.x * FLOAT_PACKING_CONSTANT),
            (int)(x.y * FLOAT_PACKING_CONSTANT),
            (int)(x.z * FLOAT_PACKING_CONSTANT),
            (int)(x.w * FLOAT_PACKING_CONSTANT)};
}

__device__ inline float4 unpack_float4(int4 x)
{
    return {x.x / FLOAT_PACKING_CONSTANT,
            x.y / FLOAT_PACKING_CONSTANT,
            x.z / FLOAT_PACKING_CONSTANT,
            x.w / FLOAT_PACKING_CONSTANT};
}

__device__ inline void accumulate_gradient(int4* gradient_buffer, uint32_t idx, float4 gradient)
{
    int4 packed = pack_float4(gradient);
    atomicAdd(&gradient_buffer[idx].x, packed.x);
    atomicAdd(&gradient_buffer[idx].y, packed.y);
    atomicAdd(&gradient_buffer[idx].z, packed.z);
    atomicAdd(&gradient_buffer[idx].w, packed.w);
}
