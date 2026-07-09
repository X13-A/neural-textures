#include "app/ui.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include <imgui.h>

#include "app/app.h"
#include "render/neural_renderer.h"

namespace
{
const ImVec4 kRed    = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
const ImVec4 kGreen  = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);
const ImVec4 kBlue   = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
const ImVec2 kButton = ImVec2(150.0f, 0.0f);

void section_padding()
{
    ImGui::Dummy(ImVec2(0.0f, ImGui::GetTextLineHeight() * 0.5f));
}

void draw_info(AppState& app)
{
    if (!ImGui::CollapsingHeader("Info", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::SeparatorText("General");
    ImGui::Text("Iteration: %u", app.stats.iteration);
    ImGui::Text("Loss:      %.5f", app.stats.loss);

    int inf_w = 0, inf_h = 0;
    renderer_get_inference_resolution(&inf_w, &inf_h);
    ImGui::Text("Inference Res: %dx%d", inf_w, inf_h);

    ImGui::SeparatorText("Profiling");
    KernelTimings timings = renderer_get_timings();
    ImGui::Text("Backprop:   %.3f ms", timings.backprop_ms);
    ImGui::Text("Optimize:   %.3f ms", timings.optimize_ms);
    ImGui::Text("HG Grad:    %.3f ms", timings.hg_gradient_ms);
    ImGui::Text("Synthesis:  %.3f ms", timings.synthesis_ms);
    ImGui::Text("Total:      %.3f ms", timings.total_ms + timings.synthesis_ms);
}

// Highlighted button that renders in the "active" colour when `active` is true
bool mode_button(const char* label, bool active)
{
    if (active) { ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive)); }
    bool pressed = ImGui::Button(label, kButton);
    if (active) { ImGui::PopStyleColor(); }
    return pressed;
}

void draw_mode_selection(AppState& app)
{
    ImGui::SeparatorText("Mode Selection");

    if (mode_button("Feed Static Texture", app.feed_mode == FeedMode::Static))
    {
        app_set_mode_static(app);
    }

    ImGui::SameLine();

    if (mode_button("Feed Live Window", app.feed_mode == FeedMode::Live))
    {
        app_set_mode_live(app);
    }

    if (app.feed_mode == FeedMode::Live && !app.live_capturing())
    {
        ImGui::SameLine();
        float h = ImGui::GetFrameHeight();
        if (ImGui::Button("R##refresh", ImVec2(h, h)))
        {
            app_refresh_windows(app);
        }
        if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Refresh windows"); }
    }
}

void draw_active_feed_header(AppState& app)
{
    if (app.feed_mode == FeedMode::None) { return; }

    std::string label;
    if (app.feed_mode == FeedMode::Static)
    {
        label = app.static_loaded ? filename_only(app.static_path) : "Static Texture";
    }
    else  // Live
    {
        label = app.live_capturing() ? app.capture_title : "Live Window";
    }
    ImGui::SeparatorText(label.c_str());
}

void draw_playback_controls(AppState& app, bool something_selected)
{
    if (something_selected)
    {
        if (ImGui::Button(app.paused ? "Resume" : "Pause", kButton))
        {
            app.paused = !app.paused;
        }
        if (app.feed_mode == FeedMode::Static)
        {
            ImGui::SameLine();
            if (ImGui::Button("Browse PNG path...")) { app_browse_and_load_texture(app); }
        }
    }

    if (app.live_capturing())
    {
        if (something_selected) { ImGui::SameLine(); }
        if (ImGui::Button("Stop Capture", kButton)) { app_stop_capture(app); }
        ImGui::TextColored(kGreen, "Capturing...");
    }

    if (app.feed_mode == FeedMode::Live && app.live_capturing())
    {
        ImGui::Checkbox("Show Captured Feed", &app.show_capture_debug);
    }
}

void draw_static_feed(AppState& app)
{
    if (!app.static_loaded)
    {
        if (ImGui::Button("Browse PNG path...")) { app_browse_and_load_texture(app); }
    }
    else
    {
        ImGui::Text("Source texture path: \"%s\"", app.static_path);
        ImGui::Checkbox("Show source texture", &app.show_capture_debug);
    }

    if (!app.static_error.empty())
    {
        ImGui::TextColored(kRed, "%s", app.static_error.c_str());
    }
}

void draw_window_picker(AppState& app)
{
    ImGui::TextColored(kBlue, "Select the window to use for training :");

    if (!app.available_windows.empty())
    {
        ImGui::Indent();
        for (int i = 0; i < (int)app.available_windows.size(); ++i)
        {
            bool selected = (app.selected_window_idx == i);
            if (ImGui::Selectable(app.available_windows[i].title.c_str(), selected))
            {
                app_start_capture(app, i);
            }
        }
        ImGui::Unindent();
    }

    if (!app.static_error.empty())
    {
        ImGui::TextColored(kRed, "%s", app.static_error.c_str());
    }
}

void draw_controls(AppState& app)
{
    if (!ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    const bool something_selected =
        (app.feed_mode == FeedMode::Static && app.static_loaded) ||
        (app.feed_mode == FeedMode::Live   && app.live_capturing());

    draw_mode_selection(app);
    draw_active_feed_header(app);
    draw_playback_controls(app, something_selected);

    if (app.feed_mode == FeedMode::Static)
    {
        draw_static_feed(app);
    }
    else if (app.feed_mode == FeedMode::Live && !app.live_capturing())
    {
        draw_window_picker(app);
    }

    if (something_selected)
    {
        ImGui::Spacing();
        if (ImGui::Button("Dump predicted vs target", ImVec2(250.0f, 0.0f))) { app_dump_comparison(app); }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Render at original resolution and save inferred image with source side by side into the \"dump\" folder");
        }
    }
}

void draw_presets(AppState& app)
{
    ImGui::SeparatorText("Presets");

    if (ImGui::Button("Static", kButton)) { app_apply_preset_static(app); }
    ImGui::SameLine();
    if (ImGui::Button("Live", kButton))   { app_apply_preset_live(app); }

    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Static: for fixed images");
    ImGui::TextDisabled("Live: for dynamic content changing in real-time (games, videos)");
    ImGui::PopTextWrapPos();
}

void draw_settings(AppState& app)
{
    ImGui::SeparatorText("Settings");

    ImGui::InputText("Learning Rate", app.lr_buffer, sizeof(app.lr_buffer));
    ImGui::SameLine();
    if (ImGui::Button("Apply##lr"))
    {
        renderer_set_learning_rate(std::strtof(app.lr_buffer, nullptr));
        renderer_reset_training();
    }

    bool use_sgd = renderer_get_use_sgd();
    if (ImGui::Checkbox("Disable Momentum (SGD)", &use_sgd))
    {
        renderer_set_use_sgd(use_sgd);
        renderer_reset_training();
    }

    if (ImGui::SliderFloat("Resolution Scale", &app.resolution_scale, 0.1f, 2.0f, "%.2f"))
    {
        renderer_set_resolution_scale(app.resolution_scale);
    }

    if (ImGui::SliderInt("Training Steps/Frame", &app.training_steps, 1, 250))
    {
        renderer_set_training_steps_per_frame(app.training_steps);
    }

    ImGui::Spacing();

    if (ImGui::Button("Reset Training", kButton)) { renderer_reset_training(); }
    ImGui::SameLine();
    if (ImGui::Checkbox("Every Frame##full", &app.reset_every_frame)) { renderer_reset_training(); }

    if (ImGui::Button("Reset Hash Grid", kButton)) { renderer_reset_hashgrid_features(); }
    ImGui::SameLine();
    if (ImGui::Checkbox("Every Frame##hg", &app.reset_hg_every_frame)) { renderer_reset_training(); }

    ImGui::Spacing();

    bool clear_adam = renderer_get_clear_adam_on_capture();
    if (ImGui::Checkbox("Clear Adam on Capture Update", &clear_adam))
    {
        renderer_set_clear_adam_on_capture(clear_adam);
        renderer_reset_training();
    }

    bool rand_rng = renderer_get_randomize_rng();
    if (ImGui::Checkbox("Randomize RNG per Frame", &rand_rng))
    {
        renderer_set_randomize_rng(rand_rng);
        renderer_reset_training();
    }
}

void draw_training_settings(AppState& app)
{
    if (!ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    draw_presets(app);
    draw_settings(app);
}

void draw_visualization(AppState& app)
{
    if (!ImGui::CollapsingHeader("Hash Grid Visualization")) { return; }

    constexpr float kCycleSeconds = 5.0f;

    bool visualize = renderer_get_visualize_hashgrid();
    if (ImGui::Checkbox("Show hash grid encoding", &visualize))
    {
        renderer_set_visualize_hashgrid(visualize);
    }

    const int channel_count = renderer_get_hashgrid_channel_count();
    ImGui::TextDisabled("Shows one of the %d encoded channels as greyscale.", channel_count);

    ImGui::BeginDisabled(!visualize);

    int channel = renderer_get_vis_channel();
    if (ImGui::SliderInt("Channel", &channel, 0, channel_count - 1))
    {
        renderer_set_vis_channel(channel);
    }
    ImGui::EndDisabled();

    ImGui::Spacing();
    if (ImGui::Button("Export All Channels...", kButton))
    {
        app_export_hashgrid_channels(app);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Save every hash-grid channel as a greyscale PNG into a chosen folder");
    }
}
}  // namespace

void ui_draw(AppState& app)
{
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.7f);
    if (ImGui::Begin("Neural Texture", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings  |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav))
    {
        draw_info(app);
        section_padding();

        draw_controls(app);
        section_padding();

        draw_training_settings(app);
        section_padding();

        draw_visualization(app);
        section_padding();
    }
    ImGui::End();
}
