#pragma once

#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <memory>
#include <functional>
#include <cstdint>

// Windows Graphics Capture session for capturing a window into a D3D11 texture
class WGCCapture
{
public:
    using FrameCallback = std::function<void(ID3D11Texture2D*)>;

    WGCCapture();
    ~WGCCapture();

    bool start_capture(HWND target_window);
    void stop_capture();

    // Call once per app frame. Invokes the callback with the captured texture if a new frame is available
    void poll_frame(const FrameCallback& callback);

    bool is_capturing() const { return is_capturing_; }

private:
    void cleanup();

    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> device_context_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> current_frame_;

    int capture_width_ = 0;
    int capture_height_ = 0;
    bool is_capturing_ = false;

    void* capture_manager_ = nullptr;  // Direct3D11CaptureFramePool
};
