#include "platform/capture/wgc_capture.h"
#include <iostream>
#include <d3d11.h>
#include <dxgi1_2.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// Context to hold duplication + window handle for position tracking
struct DuplicationContext {
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication;
    HWND target_window = nullptr;
    int window_x = 0;
    int window_y = 0;
};

WGCCapture::WGCCapture()
{
    D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    UINT feature_level_count = ARRAYSIZE(feature_levels);
    D3D_FEATURE_LEVEL out_feature_level;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        feature_levels,
        feature_level_count,
        D3D11_SDK_VERSION,
        device_.GetAddressOf(),
        &out_feature_level,
        device_context_.GetAddressOf());

    if (FAILED(hr))
    {
        std::cerr << "Failed to create D3D11 device: 0x" << std::hex << hr << std::endl;
        throw std::runtime_error("D3D11 device creation failed");
    }

    std::cout << "D3D11 device created" << std::endl;
}

WGCCapture::~WGCCapture()
{
    stop_capture();
}

bool WGCCapture::start_capture(HWND target_window)
{
    if (!IsWindow(target_window))
        return false;

    try
    {
        RECT window_rect;
        if (!GetWindowRect(target_window, &window_rect))
            return false;

        capture_width_ = window_rect.right - window_rect.left;
        capture_height_ = window_rect.bottom - window_rect.top;

        if (capture_width_ <= 0 || capture_height_ <= 0)
            return false;

        // Get DXGI device and duplication
        Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
        HRESULT hr = device_->QueryInterface(IID_PPV_ARGS(dxgi_device.GetAddressOf()));
        if (FAILED(hr)) return false;

        Microsoft::WRL::ComPtr<IDXGIAdapter> dxgi_adapter;
        hr = dxgi_device->GetAdapter(dxgi_adapter.GetAddressOf());
        if (FAILED(hr)) return false;

        Microsoft::WRL::ComPtr<IDXGIOutput> dxgi_output;
        hr = dxgi_adapter->EnumOutputs(0, dxgi_output.GetAddressOf());
        if (FAILED(hr)) return false;

        Microsoft::WRL::ComPtr<IDXGIOutput1> dxgi_output1;
        hr = dxgi_output->QueryInterface(IID_PPV_ARGS(dxgi_output1.GetAddressOf()));
        if (FAILED(hr)) return false;

        Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication;
        hr = dxgi_output1->DuplicateOutput(device_.Get(), duplication.GetAddressOf());
        if (FAILED(hr)) return false;

        // Create capture texture
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = capture_width_;
        desc.Height = capture_height_;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;

        hr = device_->CreateTexture2D(&desc, nullptr, current_frame_.GetAddressOf());
        if (FAILED(hr)) return false;

        // Store context with window handle and position for cropping
        auto ctx = new DuplicationContext();
        ctx->duplication = duplication;
        ctx->target_window = target_window;
        ctx->window_x = window_rect.left;
        ctx->window_y = window_rect.top;
        capture_manager_ = ctx;

        is_capturing_ = true;
        std::cout << "Capture started: " << capture_width_ << "x" << capture_height_
                  << " at (" << ctx->window_x << ", " << ctx->window_y << ")" << std::endl;
        return true;
    }
    catch (const std::exception&)
    {
        cleanup();
        return false;
    }
}

void WGCCapture::stop_capture()
{
    cleanup();
}

void WGCCapture::poll_frame(const FrameCallback& callback)
{
    if (!is_capturing_ || !capture_manager_)
        return;

    try
    {
        auto ctx = reinterpret_cast<DuplicationContext*>(capture_manager_);

        // Fetch current window position and size every frame
        RECT window_rect;
        if (!IsWindow(ctx->target_window) || !GetWindowRect(ctx->target_window, &window_rect))
            return;

        int new_width = window_rect.right - window_rect.left;
        int new_height = window_rect.bottom - window_rect.top;

        if (new_width <= 0 || new_height <= 0)
            return;

        // Update position and size if changed
        ctx->window_x = window_rect.left;
        ctx->window_y = window_rect.top;
        capture_width_ = new_width;
        capture_height_ = new_height;

        Microsoft::WRL::ComPtr<IDXGIResource> desktop_resource;
        DXGI_OUTDUPL_FRAME_INFO frame_info;

        HRESULT hr = ctx->duplication->AcquireNextFrame(0, &frame_info, desktop_resource.GetAddressOf());

        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
            return;  // No new frame

        if (FAILED(hr))
            return;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> desktop_texture;
        hr = desktop_resource->QueryInterface(IID_PPV_ARGS(desktop_texture.GetAddressOf()));
        if (FAILED(hr))
        {
            ctx->duplication->ReleaseFrame();
            return;
        }

        // Recreate capture texture if size changed
        D3D11_TEXTURE2D_DESC current_desc;
        current_frame_->GetDesc(&current_desc);
        if ((int)current_desc.Width != capture_width_ || (int)current_desc.Height != capture_height_)
        {
            current_frame_.Reset();
            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = capture_width_;
            desc.Height = capture_height_;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;

            HRESULT hr_create = device_->CreateTexture2D(&desc, nullptr, current_frame_.GetAddressOf());
            if (FAILED(hr_create))
            {
                ctx->duplication->ReleaseFrame();
                return;
            }
        }

        // Copy window region from desktop
        D3D11_BOX src_box;
        src_box.left = ctx->window_x;
        src_box.top = ctx->window_y;
        src_box.front = 0;
        src_box.right = ctx->window_x + capture_width_;
        src_box.bottom = ctx->window_y + capture_height_;
        src_box.back = 1;

        device_context_->CopySubresourceRegion(
            current_frame_.Get(), 0, 0, 0, 0,
            desktop_texture.Get(), 0, &src_box);

        if (callback)
            callback(current_frame_.Get());

        ctx->duplication->ReleaseFrame();
    }
    catch (const std::exception&)
    {
    }
}

void WGCCapture::cleanup()
{
    is_capturing_ = false;

    if (capture_manager_)
    {
        delete reinterpret_cast<DuplicationContext*>(capture_manager_);
        capture_manager_ = nullptr;
    }

    current_frame_.Reset();
    capture_width_ = 0;
    capture_height_ = 0;
}
