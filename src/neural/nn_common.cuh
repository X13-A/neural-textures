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

// Gradient accumulation

__device__ inline void accumulate_gradient(float4* gradient_buffer, uint32_t idx, float4 gradient)
{
    // Warp-aggregated float atomicAdd
    auto group = cg::labeled_partition(cg::coalesced_threads(), idx);

    float x = cg::reduce(group, gradient.x, cg::plus<float>());
    float y = cg::reduce(group, gradient.y, cg::plus<float>());
    float z = cg::reduce(group, gradient.z, cg::plus<float>());
    float w = cg::reduce(group, gradient.w, cg::plus<float>());

    if (group.thread_rank() == 0)
    {
        atomicAdd(&gradient_buffer[idx].x, x);
        atomicAdd(&gradient_buffer[idx].y, y);
        atomicAdd(&gradient_buffer[idx].z, z);
        atomicAdd(&gradient_buffer[idx].w, w);
    }
}
