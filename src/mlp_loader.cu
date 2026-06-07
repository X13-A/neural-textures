#include "mlp_loader.h"
#include "cuda_check.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void mlp_load_config_from_file(const char* path, MLP_Configuration* config, std::vector<float4>* host_parameters)
{
    auto throw_invalid_config_error = []() { throw std::runtime_error("Invalid configuration file"); };
    try 
    {
        std::ifstream ifs(path);
        if (!ifs)
        {
            throw std::runtime_error(std::string("Failed to open parameter file: ") + path);
        }
        json j;
        try 
        {
            ifs >> j;
        }
        catch (const std::exception& e) 
        {
            throw std::runtime_error(std::string("Failed to parse JSON: ") + e.what());
        }

        // Load and pack all parameters in single pass
        if (j["layers"].empty()) throw_invalid_config_error();

        std::vector<float4> all_host_float4s;
        size_t offset = 0;
        int layer_idx = 0;

        for (size_t li = 0; li < j["layers"].size(); ++li)
        {
            const auto& lj = j["layers"][li];
            if (!lj.contains("input_size"))                         throw_invalid_config_error();
            if (!lj.contains("output_size"))                        throw_invalid_config_error();
            if (!lj.contains("params") || !lj["params"].is_array()) throw_invalid_config_error();

            size_t input_size  = lj["input_size"].get<size_t>();
            size_t output_size = lj["output_size"].get<size_t>();
            if (input_size == 0 || output_size == 0) throw_invalid_config_error();

            // Compute quartet counts
            size_t prev_q = (input_size + 3) / 4;
            size_t curr_q = (output_size + 3) / 4;
            
            if (prev_q == 0 || curr_q == 0) throw_invalid_config_error();
            if (curr_q > MAX_LAYER_QUARTETS) 
            {
                throw std::runtime_error("Layer size exceeds maximum supported quartet: " + std::to_string(curr_q) + " > " + std::to_string(MAX_LAYER_QUARTETS));
            }

            size_t expected_float4s = curr_q * (prev_q * 4 + 1);
            size_t expected_floats = expected_float4s * 4;

            if (lj["params"].size() != expected_floats) throw_invalid_config_error();

            // Set input sizes
            if (li == 0) 
            {
                config->layer_sizes[layer_idx++] = static_cast<uint32_t>(prev_q * 4);
            }
            config->layer_sizes[layer_idx++] = static_cast<uint32_t>(curr_q * 4);

            // Set activations
            if (li == j["layers"].size() - 1)
            {
                config->activation_functions[li] = ActivationFunction::None;
            }
            else
            {
                config->activation_functions[li] = ActivationFunction::ReLU;
            }

            // Load parameters
            const auto& params = lj["params"];
            for (size_t i = 0; i < expected_float4s; ++i)
            {
                float4 v;
                v.x = params[i * 4 + 0].get<float>();
                v.y = params[i * 4 + 1].get<float>();
                v.z = params[i * 4 + 2].get<float>();
                v.w = params[i * 4 + 3].get<float>();
                all_host_float4s.push_back(v);
            }

            // Set parameter offsets
            config->parameter_offsets[li] = offset;
            offset += expected_float4s;
        }

        config->num_layers = layer_idx;
        size_t total_float4s = all_host_float4s.size();

        host_parameters->resize(total_float4s);
        std::copy(all_host_float4s.begin(), all_host_float4s.end(), host_parameters->begin());
        std::cout << "Loaded MLP config: " << config->num_layers << " layers, total parameters (float4): " << total_float4s << std::endl;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "mlp_load_config_from_file error: %s\n", e.what());
        std::fflush(stderr);
        std::abort();
    }
    catch (...) {
        std::fprintf(stderr, "mlp_load_config_from_file unknown exception\n");
        std::fflush(stderr);
        std::abort();
    }
}
