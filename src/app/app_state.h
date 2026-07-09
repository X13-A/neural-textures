#pragma once

#include <memory>
#include <string>
#include <vector>

#include "render/neural_renderer.h"
#include "platform/capture/window_picker.h"
#include "platform/capture/wgc_capture.h"

enum class FeedMode { None, Static, Live };

struct AppState
{
    int width  = 0;
    int height = 0;

    FrameStats stats;

    std::unique_ptr<WGCCapture> capture;
    std::vector<WindowInfo>     available_windows;
    int                         selected_window_idx = -1;
    std::string                 capture_title;

    FeedMode feed_mode          = FeedMode::None;
    bool     paused             = false;
    bool     show_capture_debug = false;

    char        static_path[512] = {0};
    bool        static_loaded    = false;
    std::string static_error;

    char  lr_buffer[32]        = {0};
    float resolution_scale     = 0.5f;
    int   training_steps       = 150;
    bool  reset_every_frame    = false;
    bool  reset_hg_every_frame = true;

    bool live_capturing() const { return capture && capture->is_capturing(); }
};
