#include "cuda_check.h"
#include "mlp_loader.h"
#include "mlp_inference.h"
#include "mlp_shared.h"
#include <cstring>
#include <chrono>

static Pixel* d_image = nullptr;
static int g_width = 0;
static int g_height = 0;

// MLP state
static MLP_Configuration g_mlp_config;
static MLP_Buffers g_mlp_buffers;

__global__
void synthesize(Pixel* image, const int width, const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    float u = static_cast<float>(x) / width;
    float v = static_cast<float>(y) / height;
    image[idx] = Pixel{
        static_cast<unsigned char>(u * 255),
        static_cast<unsigned char>(v * 255),
        0,
        255
    };
}

void screenspace_shader_init(int width, int height)
{
    g_width = width;
    g_height = height;
    size_t bytes = static_cast<size_t>(width) * height * sizeof(Pixel);
    CUDA_CHECK(cudaMalloc(&d_image, bytes));

    std::memset(&g_mlp_config, 0, sizeof(g_mlp_config));
    std::memset(&g_mlp_buffers, 0, sizeof(g_mlp_buffers));

    std::vector<float4> host_parameters;
    mlp_load_config_from_file("texture_mlp_parameters.json", &g_mlp_config, &host_parameters);
    mlp_validate_config(&g_mlp_config);
    mlp_load_host_parameters(host_parameters.data(), host_parameters.size(), &g_mlp_buffers);
}

void screenspace_shader_run(unsigned char* host_ptr)
{
    if (!d_image) return;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    mlp_synthesis_run(g_width, g_height, d_image, g_mlp_config, g_mlp_buffers);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    size_t bytes = static_cast<size_t>(g_width) * g_height * sizeof(Pixel);
    CUDA_CHECK(cudaMemcpy(host_ptr, d_image, bytes, cudaMemcpyDeviceToHost));
}

void screenspace_shader_shutdown()
{
    if (d_image) CUDA_CHECK(cudaFree(d_image));
    d_image = nullptr;
    mlp_free(&g_mlp_buffers);
}
