#pragma once
#include <cstdint>
#include "neural/utils/cuda_timing.h"

struct ID3D11Texture2D;

struct FrameStats
{
    float    inference_ms = 0.0f;
    float    loss         = 0.0f;
    uint32_t iteration    = 0;
};

void       renderer_init(int width, int height);
FrameStats renderer_run(unsigned char* host_ptr);
void       renderer_shutdown();

void renderer_update_target_from_d3d11(ID3D11Texture2D* texture);

// Set the training target from a host RGBA8 image
void renderer_set_target_from_rgba(const unsigned char* rgba, int width, int height);

struct CapturedFrame
{
    unsigned char* pixels = nullptr;
    int            width  = 0;
    int            height = 0;
};
CapturedFrame renderer_get_last_capture();

float renderer_get_learning_rate();
void  renderer_set_learning_rate(float lr);

void renderer_set_resolution_scale(float scale);

int  renderer_get_training_steps_per_frame();
void renderer_set_training_steps_per_frame(int steps);

void renderer_get_inference_resolution(int* out_width, int* out_height);

KernelTimings renderer_get_timings();

void renderer_reset_training();
void renderer_reset_hashgrid_features();

bool renderer_get_clear_adam_on_capture();
void renderer_set_clear_adam_on_capture(bool enabled);

bool renderer_get_randomize_rng();
void renderer_set_randomize_rng(bool enabled);

bool renderer_get_use_sgd();
void renderer_set_use_sgd(bool enabled);

bool renderer_get_visualize_hashgrid();
void renderer_set_visualize_hashgrid(bool enabled);
int  renderer_get_vis_channel();
void renderer_set_vis_channel(int channel);
int  renderer_get_hashgrid_channel_count();

void renderer_render_hashgrid_channel(int channel, int width, int height, unsigned char* out_rgba);

// Renders at an arbitrary resolution in out_rgba
void renderer_render_at_resolution(int width, int height, unsigned char* out_rgba);

