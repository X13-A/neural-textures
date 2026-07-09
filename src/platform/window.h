#pragma once

#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

class Window
{
public:
    using FrameCallback = std::function<void(unsigned char*)>;
    using ImGuiCallback = std::function<void()>;

    Window(int width, int height, const std::string& title);
    ~Window();

    void run(const FrameCallback& frameCb, const ImGuiCallback& guiCb);

private:
    void draw_frame(const unsigned char* pixels);

    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    unsigned int tex_ = 0;
    std::vector<unsigned char> host_buffer_;
};
