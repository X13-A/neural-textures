#include <cstdlib>
#include <iostream>

#include "window.h"
#include "screenspace_shader.h"

int main()
{
    constexpr int output_width = 512;
    constexpr int output_height = 512;

    try
    {
        Window win(output_width, output_height, "Neural Texture");
        screenspace_shader_init(output_width, output_height);

        // Main loop
        win.run([](unsigned char* ptr)
        {
            screenspace_shader_run(ptr);
        });

        screenspace_shader_shutdown();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}