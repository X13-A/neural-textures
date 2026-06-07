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

# Prevents nlohmann/json config error
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Allow external projects that declare older CMake minimum versions")

# nlohmann/json for parameter file parsing
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(cuda_neural_texture
    src/main.cpp
    src/window.cpp
    src/screenspace_shader.cu
    src/mlp_loader.cu
    src/mlp_inference.cu
)

find_package(OpenGL REQUIRED)
target_link_libraries(cuda_neural_texture PRIVATE glfw OpenGL::GL)
target_link_libraries(cuda_neural_texture PRIVATE nlohmann_json::nlohmann_json)

# Change for your GPU if desired
# 86 = RTX 30xx
# 89 = RTX 40xx
# 90 = Hopper
set_property(TARGET cuda_neural_texture PROPERTY
    CUDA_ARCHITECTURES 89
)