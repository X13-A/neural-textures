#pragma once
#include "mlp_shared.h"
#include <vector>

void mlp_load_config_from_file(const char* path, MLP_Configuration* config, std::vector<float4>* host_parameters);
