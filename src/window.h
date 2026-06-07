#pragma once

#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

class Window
{
public:
    using FrameCallback = std::function<void(unsigned char*)>;

    Window(int width, int height, const std::string& title);
    ~Window();

    void run(const FrameCallback& frameCb);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    unsigned int tex_ = 0;
    std::vector<unsigned char> host_buffer_;
};
