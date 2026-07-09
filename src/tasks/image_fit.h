#pragma once
#include "neural/mlp_types.h"
#include "neural/hashgrid.h"
#include "neural/utils/cuda_timing.h"

MLP_Configuration image_fit_build_mlp_config();

void image_fit_synthesis_run(
    int width, int height, uchar4* d_output,
    MLP_Configuration config, MLP_Buffers buffers,
    HashGrid_Configuration hg_config, const HashGrid_Buffers& hg_buffers,
    bool use_hashgrid = true);

// One training step: backprop + parameter update for MLP and hash grid.
// Returns average L2 loss over the batch. Timings are accumulated in out_timings.
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
    bool use_hashgrid = true,
    KernelTimings* out_timings = nullptr,
    uint32_t rng_seed = 0,
    bool use_sgd = false);
