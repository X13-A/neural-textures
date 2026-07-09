#include "platform/capture/window_picker.h"

struct EnumerateContext
{
    std::vector<WindowInfo>* windows;
};

static BOOL CALLBACK enum_window_proc(HWND hwnd, LPARAM lParam)
{
    if (!IsWindowVisible(hwnd))
        return TRUE;

    char title[256] = {};
    GetWindowTextA(hwnd, title, sizeof(title) - 1);

    if (strlen(title) > 0)
    {
        auto ctx = reinterpret_cast<EnumerateContext*>(lParam);
        ctx->windows->push_back({hwnd, title});
    }

    return TRUE;
}

std::vector<WindowInfo> enumerate_windows()
{
    std::vector<WindowInfo> windows;
    EnumerateContext ctx = {&windows};
    EnumWindows(enum_window_proc, reinterpret_cast<LPARAM>(&ctx));
    return windows;
}
