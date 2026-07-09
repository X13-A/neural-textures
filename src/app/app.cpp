#include "app/app.h"

#include <cstdio>
#include <cstring>
#include <ctime>

#include "render/neural_renderer.h"

#include <commdlg.h>
#include <shlobj.h>

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace
{
bool open_png_dialog(char* out, size_t out_size)
{
    char file[512] = {0};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "PNG Images\0*.png\0All Files\0*.*\0";
    ofn.lpstrFile   = file;
    ofn.nMaxFile    = sizeof(file);
    ofn.Flags       = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn))
    {
        std::strncpy(out, file, out_size - 1);
        out[out_size - 1] = '\0';
        return true;
    }
    return false;
}

bool open_folder_dialog(char* out)
{
    BROWSEINFOA bi = {};
    bi.lpszTitle = "Select a folder to export hash-grid channel images";
    bi.ulFlags   = BIF_RETURNONLYFSDIRS;

    LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
    if (!pidl) { return false; }

    bool ok = SHGetPathFromIDListA(pidl, out);
    ILFree(pidl);
    return ok;
}
}  // namespace

std::string filename_only(const std::string& path)
{
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

void app_init(AppState& app, int width, int height)
{
    app.width  = width;
    app.height = height;

    app.available_windows = enumerate_windows();

    std::snprintf(app.lr_buffer, sizeof(app.lr_buffer), "%.6f", renderer_get_learning_rate());
    app.training_steps = renderer_get_training_steps_per_frame();

    // Load default image
    app_apply_preset_static(app);
    app_set_mode_static(app);
    std::strncpy(app.static_path, ASSETS_DIR "/target_titus.png", sizeof(app.static_path) - 1);
    app_load_texture(app, app.static_path);
}

void app_refresh_windows(AppState& app)
{
    app.available_windows   = enumerate_windows();
    app.selected_window_idx = -1;
}

namespace
{
void apply_learning_rate(AppState& app, float lr)
{
    renderer_set_learning_rate(lr);
    std::snprintf(app.lr_buffer, sizeof(app.lr_buffer), "%.6f", lr);
}
}  // namespace

void app_apply_preset_static(AppState& app)
{
    app.reset_every_frame    = false;
    app.reset_hg_every_frame = false;
    renderer_set_clear_adam_on_capture(false);
    renderer_set_use_sgd(false);
    renderer_set_randomize_rng(true);
    apply_learning_rate(app, 1e-3f);
    app.resolution_scale = 1.0f;
    renderer_set_resolution_scale(app.resolution_scale);
    renderer_reset_training();
}

void app_apply_preset_live(AppState& app)
{
    app.reset_every_frame    = false;
    app.reset_hg_every_frame = true;
    renderer_set_clear_adam_on_capture(true);
    renderer_set_use_sgd(false);
    renderer_set_randomize_rng(true);
    apply_learning_rate(app, 1e-3f);
    app.resolution_scale = 0.5f;
    renderer_set_resolution_scale(app.resolution_scale);
    renderer_reset_training();
}

void app_set_mode_static(AppState& app)
{
    if (app.live_capturing()) { app.capture->stop_capture(); }
    app.feed_mode          = FeedMode::Static;
    app.show_capture_debug = false;
    app.paused             = false;
    app_apply_preset_static(app);
}

void app_set_mode_live(AppState& app)
{
    app.feed_mode    = FeedMode::Live;
    app.static_loaded = false;
    app.static_error.clear();
    app.paused       = false;
    app_refresh_windows(app);
    app_apply_preset_live(app);
}

bool app_load_texture(AppState& app, const char* path)
{
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(path, &w, &h, &n, 4);
    if (!data)
    {
        const char* reason = stbi_failure_reason();
        app.static_error  = std::string("Failed to load image: ") +
                            (reason ? reason : "unknown error");
        app.static_loaded = false;
        return false;
    }

    renderer_set_target_from_rgba(data, w, h);
    stbi_image_free(data);

    app.static_error.clear();
    app.static_loaded = true;
    app.paused        = false;
    renderer_reset_training();
    return true;
}

bool app_browse_and_load_texture(AppState& app)
{
    if (!open_png_dialog(app.static_path, sizeof(app.static_path))) { return false; }
    return app_load_texture(app, app.static_path);
}

bool app_export_hashgrid_channels(AppState& app)
{
    char folder[MAX_PATH] = {0};
    if (!open_folder_dialog(folder)) { return false; }

    CapturedFrame source = renderer_get_last_capture();
    const int channels = renderer_get_hashgrid_channel_count();
    const int w = source.width;
    const int h = source.height;

    std::vector<unsigned char> pixels((size_t)w * h * 4);

    // (╯°□°）╯︵ ┻━┻
    stbi_flip_vertically_on_write(1);

    bool ok = true;
    for (int c = 0; c < channels; ++c)
    {
        renderer_render_hashgrid_channel(c, w, h, pixels.data());

        char path[MAX_PATH + 64];
        std::snprintf(path, sizeof(path), "%s\\hashgrid_channel_%02d.png", folder, c);
        if (!stbi_write_png(path, w, h, 4, pixels.data(), w * 4)) { ok = false; }
    }

    stbi_flip_vertically_on_write(0);
    return ok;
}

bool app_dump_comparison(AppState& app)
{
    CapturedFrame source = renderer_get_last_capture();
    if (!source.pixels || source.width < 1 || source.height < 1)
    {
        app.static_error = "Nothing to dump: no source image loaded";
        return false;
    }

    const int w = source.width;
    const int h = source.height;

    std::vector<unsigned char> rendered((size_t)w * h * 4);
    renderer_render_at_resolution(w, h, rendered.data());

    // Flip
    std::vector<unsigned char> rendered_flipped((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
    {
        std::memcpy(rendered_flipped.data() + (size_t)y * w * 4,
                    rendered.data() + (size_t)(h - 1 - y) * w * 4,
                    (size_t)w * 4);
    }

    CreateDirectoryA("dump", nullptr);

    std::time_t now = std::time(nullptr);
    std::tm     tm{};
    localtime_s(&tm, &now);

    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);

    char source_path[MAX_PATH];
    char render_path[MAX_PATH];
    std::snprintf(source_path, sizeof(source_path), "dump/%s_source.png", stamp);
    std::snprintf(render_path, sizeof(render_path), "dump/%s_render.png", stamp);

    bool ok = stbi_write_png(source_path, w, h, 4, source.pixels, w * 4) != 0;
    ok      = stbi_write_png(render_path, w, h, 4, rendered_flipped.data(), w * 4) != 0 && ok;

    if (!ok) { app.static_error = "Failed to write dump image"; }
    return ok;
}

bool app_start_capture(AppState& app, int window_idx)
{
    if (window_idx < 0 || window_idx >= (int)app.available_windows.size()) { return false; }

    app.selected_window_idx = window_idx;
    if (!app.capture)
    {
        app.capture = std::make_unique<WGCCapture>();
    }

    if (app.capture->start_capture(app.available_windows[window_idx].hwnd))
    {
        app.capture_title = app.available_windows[window_idx].title;
        return true;
    }

    app.selected_window_idx = -1;
    app.static_error        = "Capture failed for selected window";
    return false;
}

void app_stop_capture(AppState& app)
{
    if (app.capture) { app.capture->stop_capture(); }
    app.show_capture_debug = false;
    app.capture_title.clear();
    app_refresh_windows(app);
}

void app_update_frame(AppState& app, unsigned char* host_ptr)
{
    const int W = app.width;
    const int H = app.height;

    const bool live_capturing = app.live_capturing();

    if (live_capturing)
    {
        app.capture->poll_frame([&](ID3D11Texture2D* frame_tex)
        {
            renderer_update_target_from_d3d11(frame_tex);
        });
    }

    const bool active = !app.paused &&
        ((app.feed_mode == FeedMode::Static && app.static_loaded) ||
         (app.feed_mode == FeedMode::Live   && live_capturing));

    if (app.show_capture_debug && app.feed_mode != FeedMode::None)
    {
        // Show the raw feed
        CapturedFrame frame = renderer_get_last_capture();
        if (frame.pixels && frame.width > 0 && frame.height > 0)
        {
            if (frame.width == W && frame.height == H)
            {
                // Flip before display
                for (int y = 0; y < H; ++y)
                {
                    int sy = H - 1 - y;
                    std::memcpy(host_ptr + (size_t)y * W * 4,
                                frame.pixels + (size_t)sy * W * 4,
                                (size_t)W * 4);
                }
            }
            else
            {
                for (int y = 0; y < H; ++y)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        int sx = (x * frame.width) / W;
                        int sy = ((H - 1 - y) * frame.height) / H;
                        sx = (sx >= frame.width)  ? frame.width  - 1 : sx;
                        sy = (sy >= frame.height) ? frame.height - 1 : sy;

                        int src = (sy * frame.width + sx) * 4;
                        int dst = (y * W + x) * 4;
                        host_ptr[dst + 0] = frame.pixels[src + 0];
                        host_ptr[dst + 1] = frame.pixels[src + 1];
                        host_ptr[dst + 2] = frame.pixels[src + 2];
                        host_ptr[dst + 3] = frame.pixels[src + 3];
                    }
                }
            }
        }
    }
    else if (active)
    {
        if (app.reset_every_frame)    { renderer_reset_training(); }
        if (app.reset_hg_every_frame) { renderer_reset_hashgrid_features(); }
        app.stats = renderer_run(host_ptr);
    }
    else if (!app.paused)
    {
        // Idle
        std::memset(host_ptr, 0, (size_t)W * H * 4);
    }
}
