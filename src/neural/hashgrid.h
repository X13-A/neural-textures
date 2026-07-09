#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Matches the GPU Zen 4 (https://github.com/boksajak/MLPZen) configuration
// TODO: make this configurable like in the MLP
static constexpr uint32_t HG_NUM_LEVELS      = 4;
static constexpr uint32_t HG_FEATURE_SIZE    = 4;    // Must be a multiple of 4
static constexpr uint32_t HG_BASE_RESOLUTION = 16;   // Resolution of level 0, doubles each level
static constexpr uint32_t HG_HASH_THRESHOLD  = 4096; // Max feature vectors per level before hashing

static constexpr uint32_t HG_FEATURE_QUARTETS = HG_FEATURE_SIZE / 4;
static constexpr uint32_t HG_INPUT_QUARTETS   = HG_NUM_LEVELS * HG_FEATURE_QUARTETS;
static constexpr uint32_t HG_INPUT_SIZE       = HG_NUM_LEVELS * HG_FEATURE_SIZE;

struct HashGrid_Configuration
{
    uint32_t resolutions[HG_NUM_LEVELS];
    uint32_t table_sizes[HG_NUM_LEVELS];    // resolution*resolution or HG_HASH_THRESHOLD
    uint32_t entry_offsets[HG_NUM_LEVELS];  // offsets into d_features, in float4 units
    uint32_t total_float4s;
};

struct HashGrid_Buffers
{
    float4* d_features      = nullptr;
    int4*   d_gradient      = nullptr;  // Packed integer gradient
    float4* d_adam_mean     = nullptr;
    float4* d_adam_variance = nullptr;
};

HashGrid_Configuration hg_build_config();

void hg_visualize_run(int width, int height, uchar4* d_output,
                      const HashGrid_Configuration& config, const HashGrid_Buffers& buffers,
                      int channel);

void hg_allocate(const HashGrid_Configuration& config, HashGrid_Buffers* buffers);
void hg_init_features(const HashGrid_Configuration& config, HashGrid_Buffers* buffers, uint32_t seed = 42u);

void hg_optimize(const HashGrid_Configuration& config, HashGrid_Buffers* buffers,
                 float learning_rate, float rcp_batch_size, uint32_t iteration,
                 bool use_sgd = false);

void hg_reset_training(const HashGrid_Configuration& config, HashGrid_Buffers* buffers, uint32_t seed = 42u);

// Clears Adam first/second moment buffers
void hg_clear_adam(const HashGrid_Configuration& config, HashGrid_Buffers* buffers);

void hg_free(HashGrid_Buffers* buffers);
