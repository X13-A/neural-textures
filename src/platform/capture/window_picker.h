#pragma once

#include <windows.h>
#include <string>
#include <vector>

struct WindowInfo
{
    HWND hwnd;
    std::string title;
};

// Enumerate all visible top-level windows.
std::vector<WindowInfo> enumerate_windows();
