#include "render/neural_renderer.h"
#include "neural/utils/cuda_check.h"
#include "neural/mlp.h"
#include "neural/hashgrid.h"
#include "tasks/image_fit.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <d3d11.h>
#include <wrl/client.h>

static constexpr int kBatchSize = 4 * 1024;

static float    kLearningRate           = 1e-3f;
static float    kResolutionScale        = 0.5f;
static int      kTrainingStepsPerFrame  = 150;
static bool     g_clear_adam_on_capture = true;
static bool     g_randomize_rng         = true;
static uint32_t g_rng_frame             = 0;

static bool     g_use_sgd               = false;

static bool     g_visualize_hashgrid    = false;
static int      g_vis_channel           = 0;

static uchar4*   d_output_image     = nullptr;
static uchar4*   d_target_image     = nullptr;
static int       g_target_width     = 0;
static int       g_target_height    = 0;
static int       g_width            = 0;
static int       g_height           = 0;
static uint32_t  g_iteration        = 0;
static int       g_inference_width  = 0;
static int       g_inference_height = 0;

static std::vector<unsigned char> g_last_capture_pixels;
static int g_last_capture_width  = 0;
static int g_last_capture_height = 0;

static HashGrid_Configuration g_hg_config;
static HashGrid_Buffers       g_hg_buffers;
static MLP_Configuration      g_mlp_config;
static MLP_Buffers            g_mlp_buffers;

static cudaEvent_t g_ev_start = nullptr;
static cudaEvent_t g_ev_stop  = nullptr;

static KernelTimings g_frame_timings = {};


void renderer_init(int width, int height)
{
    g_width  = width;
    g_height = height;

    // Placeholder
    g_target_width  = 16;
    g_target_height = 16;
    size_t placeholder_bytes = (size_t)g_target_width * g_target_height * 4;
    CUDA_CHECK(cudaMalloc(&d_target_image, placeholder_bytes));
    CUDA_CHECK(cudaMemset(d_target_image, 0, placeholder_bytes));

    g_hg_config  = hg_build_config();
    g_mlp_config = image_fit_build_mlp_config();
    mlp_validate_config(&g_mlp_config);

    hg_allocate(g_hg_config, &g_hg_buffers);
    hg_init_features(g_hg_config, &g_hg_buffers);

    mlp_allocate_parameters(g_mlp_config, &g_mlp_buffers);
    mlp_init_he_weights(g_mlp_config, &g_mlp_buffers);
    mlp_allocate_training_buffers(&g_mlp_buffers);

    CUDA_CHECK(cudaEventCreate(&g_ev_start));
    CUDA_CHECK(cudaEventCreate(&g_ev_stop));

    std::cout << "Neural renderer initialised. Waiting for capture input." << std::endl;
}

FrameStats renderer_run(unsigned char* host_ptr)
{
    FrameStats stats;
    stats.iteration = g_iteration;

    g_frame_timings = {};

    uint32_t rng_seed = g_randomize_rng ? g_rng_frame : 0;
    ++g_rng_frame;

    for (int step = 0; step < kTrainingStepsPerFrame; ++step)
    {
        KernelTimings step_timings = {};
        stats.loss = image_fit_train_step(
            d_target_image, g_target_width, g_target_height,
            kBatchSize, kLearningRate, g_iteration,
            g_mlp_config, g_mlp_buffers,
            g_hg_config, g_hg_buffers, true, &step_timings, rng_seed,
            g_use_sgd);

        g_frame_timings.backprop_ms    += step_timings.backprop_ms;
        g_frame_timings.optimize_ms    += step_timings.optimize_ms;
        g_frame_timings.hg_gradient_ms += step_timings.hg_gradient_ms;

        ++g_iteration;
    }

    g_frame_timings.total_ms = g_frame_timings.backprop_ms
                             + g_frame_timings.optimize_ms
                             + g_frame_timings.hg_gradient_ms;

    int inf_w = (int)(g_width  * kResolutionScale);
    int inf_h = (int)(g_height * kResolutionScale);
    inf_w = (inf_w < 1) ? 1 : inf_w;
    inf_h = (inf_h < 1) ? 1 : inf_h;

    if (inf_w != g_inference_width || inf_h != g_inference_height)
    {
        if (d_output_image)
        {
            CUDA_CHECK(cudaFree(d_output_image));
        }
        CUDA_CHECK(cudaMalloc(&d_output_image, (size_t)inf_w * inf_h * sizeof(uchar4)));
        g_inference_width  = inf_w;
        g_inference_height = inf_h;
    }

    CUDA_CHECK(cudaEventRecord(g_ev_start));
    if (g_visualize_hashgrid)
    {
        hg_visualize_run(inf_w, inf_h, d_output_image, g_hg_config, g_hg_buffers, g_vis_channel);
    }
    else
    {
        image_fit_synthesis_run(inf_w, inf_h, d_output_image, g_mlp_config, g_mlp_buffers, g_hg_config, g_hg_buffers, true);
    }
    CUDA_CHECK(cudaEventRecord(g_ev_stop));
    CUDA_CHECK(cudaEventSynchronize(g_ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&stats.inference_ms, g_ev_start, g_ev_stop));
    g_frame_timings.synthesis_ms = stats.inference_ms;

    size_t inf_bytes = (size_t)inf_w * inf_h * sizeof(uchar4);
    std::vector<unsigned char> inf_buffer(inf_bytes);
    CUDA_CHECK(cudaMemcpy(inf_buffer.data(), d_output_image, inf_bytes, cudaMemcpyDeviceToHost));

    std::memset(host_ptr, 0, (size_t)g_width * g_height * sizeof(uchar4));

    // Fit inference image into display window, preserving aspect ratio
    // FIXME
    int disp_w = inf_w;
    int disp_h = inf_h;
    if (inf_w != g_width || inf_h != g_height)
    {
        float scale = fminf((float)g_width / inf_w, (float)g_height / inf_h);
        disp_w = (int)(inf_w * scale);
        disp_h = (int)(inf_h * scale);
    }

    int off_x = (g_width  - disp_w) / 2;
    int off_y = (g_height - disp_h) / 2;

    if (disp_w == inf_w && disp_h == inf_h)
    {
        size_t row_bytes = (size_t)disp_w * sizeof(uchar4);
        for (int y = 0; y < disp_h; ++y)
        {
            std::memcpy(
                host_ptr + ((off_y + y) * g_width + off_x) * sizeof(uchar4),
                inf_buffer.data() + (size_t)y * row_bytes,
                row_bytes);
        }
    }
    else
    {
        for (int y = 0; y < disp_h; ++y)
        {
            for (int x = 0; x < disp_w; ++x)
            {
                int sx = (x * inf_w) / disp_w;
                int sy = (y * inf_h) / disp_h;
                sx = (sx >= inf_w)  ? inf_w  - 1 : sx;
                sy = (sy >= inf_h) ? inf_h - 1 : sy;

                int src = (sy * inf_w + sx) * 4;
                int dst = ((off_y + y) * g_width + (off_x + x)) * 4;

                host_ptr[dst + 0] = inf_buffer[src + 0];
                host_ptr[dst + 1] = inf_buffer[src + 1];
                host_ptr[dst + 2] = inf_buffer[src + 2];
                host_ptr[dst + 3] = inf_buffer[src + 3];
            }
        }
    }

    return stats;
}

void renderer_shutdown()
{
    if (g_ev_start) { cudaEventDestroy(g_ev_start); g_ev_start = nullptr; }
    if (g_ev_stop)  { cudaEventDestroy(g_ev_stop);  g_ev_stop  = nullptr; }

    if (d_output_image) { CUDA_CHECK(cudaFree(d_output_image)); d_output_image = nullptr; }
    if (d_target_image) { CUDA_CHECK(cudaFree(d_target_image)); d_target_image = nullptr; }

    hg_free(&g_hg_buffers);
    mlp_free(&g_mlp_buffers);
}

void renderer_update_target_from_d3d11(ID3D11Texture2D* texture)
{
    if (!texture)
    {
        std::cerr << "renderer_update_target_from_d3d11: null texture\n";
        return;
    }

    try
    {
        Microsoft::WRL::ComPtr<ID3D11Device> device;
        texture->GetDevice(device.GetAddressOf());

        Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
        device->GetImmediateContext(context.GetAddressOf());

        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);

        int src_w = (int)desc.Width;
        int src_h = (int)desc.Height;

        if (src_w <= 0 || src_h <= 0)
        {
            std::cerr << "renderer_update_target_from_d3d11: invalid dimensions " << src_w << "x" << src_h << "\n";
            return;
        }

        // Reallocate target buffer if dimensions changed
        if (src_w != g_target_width || src_h != g_target_height)
        {
            CUDA_CHECK(cudaFree(d_target_image));
            g_target_width  = src_w;
            g_target_height = src_h;
            CUDA_CHECK(cudaMalloc(&d_target_image, (size_t)src_w * src_h * sizeof(uchar4)));
            std::cout << "Capture target resized to " << src_w << "x" << src_h << "\n";
        }

        D3D11_TEXTURE2D_DESC staging_desc = desc;
        staging_desc.Usage          = D3D11_USAGE_STAGING;
        staging_desc.BindFlags      = 0;
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
        HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, staging.GetAddressOf());
        if (FAILED(hr))
        {
            std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << "\n";
            return;
        }

        context->CopyResource(staging.Get(), texture);

        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr))
        {
            std::cerr << "Failed to map staging texture: 0x" << std::hex << hr << "\n";
            return;
        }

        // Copy row by row
        std::vector<uint8_t> host_pixels((size_t)src_w * src_h * 4);
        const auto* src = reinterpret_cast<const uint8_t*>(mapped.pData);
        for (int y = 0; y < src_h; ++y)
        {
            std::memcpy(
                host_pixels.data() + (size_t)y * src_w * 4,
                src + (size_t)y * mapped.RowPitch,
                (size_t)src_w * 4);
        }

        context->Unmap(staging.Get(), 0);

        // Convert from BGRA to RGBA
        std::vector<uchar4> rgba((size_t)src_w * src_h);
        for (int i = 0; i < src_w * src_h; ++i)
        {
            rgba[i] = make_uchar4(host_pixels[i*4+2], host_pixels[i*4+1], host_pixels[i*4+0], host_pixels[i*4+3]);
        }

        CUDA_CHECK(cudaMemcpy(d_target_image, rgba.data(), rgba.size() * sizeof(uchar4), cudaMemcpyHostToDevice));

        if (g_clear_adam_on_capture)
        {
            mlp_clear_adam(&g_mlp_buffers);
            hg_clear_adam(g_hg_config, &g_hg_buffers);
        }

        // Cache last frame for debug display
        g_last_capture_pixels.resize((size_t)src_w * src_h * 4);
        for (int i = 0; i < src_w * src_h; ++i)
        {
            g_last_capture_pixels[i*4+0] = rgba[i].x;
            g_last_capture_pixels[i*4+1] = rgba[i].y;
            g_last_capture_pixels[i*4+2] = rgba[i].z;
            g_last_capture_pixels[i*4+3] = rgba[i].w;
        }
        g_last_capture_width  = src_w;
        g_last_capture_height = src_h;
    }
    catch (const std::exception& e)
    {
        std::cerr << "renderer_update_target_from_d3d11: " << e.what() << "\n";
    }
}

void renderer_set_target_from_rgba(const unsigned char* rgba, int width, int height)
{
    if (!rgba || width <= 0 || height <= 0)
    {
        std::cerr << "renderer_set_target_from_rgba: invalid input\n";
        return;
    }

    if (width != g_target_width || height != g_target_height)
    {
        CUDA_CHECK(cudaFree(d_target_image));
        g_target_width  = width;
        g_target_height = height;
        CUDA_CHECK(cudaMalloc(&d_target_image, (size_t)width * height * sizeof(uchar4)));
    }

    CUDA_CHECK(cudaMemcpy(d_target_image, rgba,
                          (size_t)width * height * sizeof(uchar4), cudaMemcpyHostToDevice));

    if (g_clear_adam_on_capture)
    {
        mlp_clear_adam(&g_mlp_buffers);
        hg_clear_adam(g_hg_config, &g_hg_buffers);
    }

    // Cache for debug display
    g_last_capture_pixels.assign(rgba, rgba + (size_t)width * height * 4);
    g_last_capture_width  = width;
    g_last_capture_height = height;
}

CapturedFrame renderer_get_last_capture()
{
    CapturedFrame frame = {};
    if (!g_last_capture_pixels.empty())
    {
        frame.pixels = g_last_capture_pixels.data();
        frame.width  = g_last_capture_width;
        frame.height = g_last_capture_height;
    }
    return frame;
}

float renderer_get_learning_rate() { return kLearningRate; }

void renderer_set_learning_rate(float lr)
{
    if (lr > 0.0f)
    {
        kLearningRate = lr;
    }
}

void renderer_set_resolution_scale(float scale)
{
    if (scale >= 0.1f && scale <= 2.0f)
    {
        kResolutionScale = scale;
    }
}

int renderer_get_training_steps_per_frame() { return kTrainingStepsPerFrame; }

void renderer_set_training_steps_per_frame(int steps)
{
    if (steps >= 1 && steps <= 250)
    {
        kTrainingStepsPerFrame = steps;
    }
}

void renderer_get_inference_resolution(int* out_width, int* out_height)
{
    if (out_width)  { *out_width  = g_inference_width;  }
    if (out_height) { *out_height = g_inference_height; }
}

KernelTimings renderer_get_timings() { return g_frame_timings; }

void renderer_reset_training()
{
    uint32_t seed = g_randomize_rng ? g_rng_frame : 0;
    mlp_reset_training(g_mlp_config, &g_mlp_buffers, seed);
    hg_reset_training(g_hg_config, &g_hg_buffers, seed);
    g_iteration = 0;
}

void renderer_reset_hashgrid_features()
{
    uint32_t seed = g_randomize_rng ? g_rng_frame : 42u;
    hg_init_features(g_hg_config, &g_hg_buffers, seed);
}

bool renderer_get_clear_adam_on_capture()         { return g_clear_adam_on_capture; }
void renderer_set_clear_adam_on_capture(bool e)   { g_clear_adam_on_capture = e; }

bool renderer_get_randomize_rng()                 { return g_randomize_rng; }
void renderer_set_randomize_rng(bool e)           { g_randomize_rng = e; }

bool renderer_get_use_sgd()                       { return g_use_sgd; }
void renderer_set_use_sgd(bool e)                 { g_use_sgd = e; }

bool renderer_get_visualize_hashgrid()             { return g_visualize_hashgrid; }
void renderer_set_visualize_hashgrid(bool e)       { g_visualize_hashgrid = e; }

int  renderer_get_vis_channel() { return g_vis_channel; }

void renderer_set_vis_channel(int channel)
{
    int max_ch = (int)HG_INPUT_SIZE - 1;
    g_vis_channel = (channel < 0) ? 0 : (channel > max_ch ? max_ch : channel);
}

int renderer_get_hashgrid_channel_count() { return (int)HG_INPUT_SIZE; }

void renderer_render_hashgrid_channel(int channel, int width, int height, unsigned char* out_rgba)
{
    if (width < 1 || height < 1 || !out_rgba) { return; }

    uchar4* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)width * height * sizeof(uchar4)));
    hg_visualize_run(width, height, d_tmp, g_hg_config, g_hg_buffers, channel);
    CUDA_CHECK(cudaMemcpy(out_rgba, d_tmp, (size_t)width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp));
}

void renderer_render_at_resolution(int width, int height, unsigned char* out_rgba)
{
    if (width < 1 || height < 1 || !out_rgba) { return; }

    uchar4* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)width * height * sizeof(uchar4)));
    image_fit_synthesis_run(width, height, d_tmp, g_mlp_config, g_mlp_buffers, g_hg_config, g_hg_buffers, true);
    CUDA_CHECK(cudaMemcpy(out_rgba, d_tmp, (size_t)width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tmp));
}

