#pragma once

void screenspace_shader_init(int width, int height);
void screenspace_shader_run(unsigned char* host_ptr);
void screenspace_shader_shutdown();
