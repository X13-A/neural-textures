#include <cstdlib>
#include <exception>
#include <iostream>

#include "platform/window.h"
#include "render/neural_renderer.h"
#include "app/app.h"
#include "app/ui.h"

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        std::cerr << "Usage: " << argv[0] << "\n";
        return EXIT_FAILURE;
    }

    constexpr int kWidth  = 1024;
    constexpr int kHeight = 1024;

    try
    {
        Window win(kWidth, kHeight, "Neural Texture");
        renderer_init(kWidth, kHeight);

        AppState app;
        app_init(app, kWidth, kHeight);

        win.run(
            [&](unsigned char* ptr) { app_update_frame(app, ptr); },
            [&]()                    { ui_draw(app); });

        renderer_shutdown();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
