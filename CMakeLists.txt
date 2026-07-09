cmake_minimum_required(VERSION 3.25)

project(cuda_neural_texture LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

add_executable(cuda_neural_texture
    src/main.cpp
    src/app/app.cpp
    src/app/ui.cpp
    src/platform/window.cpp
    src/platform/capture/window_picker.cpp
    src/platform/capture/wgc_capture.cpp
    src/render/neural_renderer.cu
    src/neural/mlp.cu
    src/neural/hashgrid.cu
    src/tasks/image_fit.cu
)

find_package(OpenGL REQUIRED)
target_link_libraries(cuda_neural_texture PRIVATE glfw OpenGL::GL)
target_link_libraries(cuda_neural_texture PRIVATE imgui_lib)

target_include_directories(cuda_neural_texture PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(cuda_neural_texture PRIVATE ${stb_SOURCE_DIR})

# Absolute path to assets
target_compile_definitions(cuda_neural_texture PRIVATE ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
target_link_libraries(cuda_neural_texture PRIVATE d3d11 dxgi windowsapp comdlg32 shell32 ole32)
target_compile_options(cuda_neural_texture PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/await>)

# Target architecture
set_property(TARGET cuda_neural_texture PROPERTY CUDA_ARCHITECTURES 89)
