#pragma once

#include <string>

#include "app/app_state.h"


void app_init(AppState& app, int width, int height);
void app_update_frame(AppState& app, unsigned char* host_ptr);

// UI-related
void app_set_mode_static(AppState& app);
void app_set_mode_live(AppState& app);
void app_refresh_windows(AppState& app);

// Training presets
void app_apply_preset_static(AppState& app);
void app_apply_preset_live(AppState& app);

// Load texture
bool app_load_texture(AppState& app, const char* path);

// Load texture through file explorer
bool app_browse_and_load_texture(AppState& app);

// Lets the user select a folder picker and exports every hash-grid channel as a greyscale PNG into it
bool app_export_hashgrid_channels(AppState& app);

// Renders the network at the source image's native resolution
// Write both the render and the source image in the dump/ folder
bool app_dump_comparison(AppState& app);

// Starts capturing the window and trains on it
bool app_start_capture(AppState& app, int window_idx);
void app_stop_capture(AppState& app);

// Extract file name from path
std::string filename_only(const std::string& path);
