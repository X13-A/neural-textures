#pragma once
#include "neural/hashgrid.h"
#include "neural/nn_common.cuh"

// PCG-style hash for 2D integer grid coordinates
__device__ inline uint32_t hg_hash(uint32_t x, uint32_t y)
{
    uint32_t h = x * 2654435761u ^ y * 805459861u;
    h ^= h >> 17;
    h *= 0xbf324c81u;
    h ^= h >> 11;
    return h;
}

// Direct mapping for small levels, spatial hashing for large ones
__device__ inline uint32_t hg_entry_index(const HashGrid_Configuration& config, uint32_t level, uint32_t x, uint32_t y)
{
    uint32_t res = config.resolutions[level];
    if (config.table_sizes[level] < res * res)
    {
        return hg_hash(x, y) % config.table_sizes[level];
    }
    else
    {
        return y * res + x;
    }
}

__device__ inline float4 hg_fetch(
    const HashGrid_Configuration& config,
    const float4* features,
    uint32_t level, uint32_t x, uint32_t y, uint32_t q)
{
    uint32_t entry = hg_entry_index(config, level, x, y);
    return features[config.entry_offsets[level] + entry * HG_FEATURE_QUARTETS + q];
}

// Bilinearly interpolates and concatenates feature vectors from all levels
__device__ inline void hg_encode(
    const HashGrid_Configuration& config,
    const float4* features,
    float2 input,
    float4 encoded[HG_INPUT_QUARTETS])
{
    for (uint32_t level = 0; level < HG_NUM_LEVELS; level++)
    {
        float max_coord = (float)(config.resolutions[level] - 1);
        float fx = input.x * max_coord;
        float fy = input.y * max_coord;

        uint32_t x0 = (uint32_t)floorf(fx);
        uint32_t y0 = (uint32_t)floorf(fy);
        uint32_t x1 = min(x0 + 1, config.resolutions[level] - 1);
        uint32_t y1 = min(y0 + 1, config.resolutions[level] - 1);

        float wx = fx - (float)x0;
        float wy = fy - (float)y0;
        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = (1.0f - wx) * wy;
        float w10 = wx * (1.0f - wy);
        float w11 = wx * wy;

        for (uint32_t q = 0; q < HG_FEATURE_QUARTETS; q++)
        {
            float4 a = hg_fetch(config, features, level, x0, y0, q);
            float4 b = hg_fetch(config, features, level, x0, y1, q);
            float4 c = hg_fetch(config, features, level, x1, y0, q);
            float4 d = hg_fetch(config, features, level, x1, y1, q);

            float res_x = w00*a.x + w01*b.x + w10*c.x + w11*d.x;
            float res_y = w00*a.y + w01*b.y + w10*c.y + w11*d.y;
            float res_z = w00*a.z + w01*b.z + w10*c.z + w11*d.z;
            float res_w = w00*a.w + w01*b.w + w10*c.w + w11*d.w;

            encoded[level * HG_FEATURE_QUARTETS + q] = {res_x, res_y, res_z, res_w};
        }
    }
}

// Distributes input_delta back to the 4 nearest grid cells weighted by bilinear coefficients
__device__ inline void hg_accumulate_gradient(
    const HashGrid_Configuration& config,
    int4* hg_gradient,
    float2 input,
    const float4* input_delta,
    uint32_t level)
{
    float max_coord = (float)(config.resolutions[level] - 1);
    float fx = input.x * max_coord;
    float fy = input.y * max_coord;

    uint32_t x0 = (uint32_t)floorf(fx);
    uint32_t y0 = (uint32_t)floorf(fy);
    uint32_t x1 = min(x0 + 1, config.resolutions[level] - 1);
    uint32_t y1 = min(y0 + 1, config.resolutions[level] - 1);

    float wx = fx - (float)x0;
    float wy = fy - (float)y0;
    float w00 = (1.0f - wx) * (1.0f - wy);
    float w01 = (1.0f - wx) * wy;
    float w10 = wx * (1.0f - wy);
    float w11 = wx * wy;

    for (uint32_t q = 0; q < HG_FEATURE_QUARTETS; q++)
    {
        float4 delta = input_delta[level * HG_FEATURE_QUARTETS + q];

        auto accumulate = [&](uint32_t x, uint32_t y, float w) {
            uint32_t entry = hg_entry_index(config, level, x, y);
            uint32_t idx   = config.entry_offsets[level] + entry * HG_FEATURE_QUARTETS + q;
            accumulate_gradient(hg_gradient, idx, {delta.x*w, delta.y*w, delta.z*w, delta.w*w});
        };

        accumulate(x0, y0, w00);
        accumulate(x0, y1, w01);
        accumulate(x1, y0, w10);
        accumulate(x1, y1, w11);
    }
}
