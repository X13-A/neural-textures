#pragma once

struct KernelTimings
{
    float backprop_ms    = 0.0f;
    float optimize_ms    = 0.0f;
    float hg_gradient_ms = 0.0f;
    float synthesis_ms   = 0.0f;
    float total_ms       = 0.0f;
};
