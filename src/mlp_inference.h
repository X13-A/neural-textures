#pragma once
#include "mlp_shared.h"

void mlp_validate_config(MLP_Configuration* config);
void mlp_load_host_parameters(float4* host_parameters, size_t host_parameters_count, MLP_Buffers* buffers);
void mlp_free(MLP_Buffers* buffers);
void mlp_synthesis_run(int width, int height, Pixel* d_output_image, MLP_Configuration config, MLP_Buffers buffers);
