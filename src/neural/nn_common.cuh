#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>

namespace cg = cooperative_groups;

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

// Gradient accumulation in fixed-point:
// floats are scaled by FLOAT_PACKING_CONSTANT and truncated to int
static constexpr float FLOAT_PACKING_CONSTANT = 65536.0f;

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

    auto group = cg::labeled_partition(cg::coalesced_threads(), idx);

    int x = cg::reduce(group, packed.x, cg::plus<int>());
    int y = cg::reduce(group, packed.y, cg::plus<int>());
    int z = cg::reduce(group, packed.z, cg::plus<int>());
    int w = cg::reduce(group, packed.w, cg::plus<int>());

    if (group.thread_rank() == 0)
    {
        atomicAdd(&gradient_buffer[idx].x, x);
        atomicAdd(&gradient_buffer[idx].y, y);
        atomicAdd(&gradient_buffer[idx].z, z);
        atomicAdd(&gradient_buffer[idx].w, w);
    }
}