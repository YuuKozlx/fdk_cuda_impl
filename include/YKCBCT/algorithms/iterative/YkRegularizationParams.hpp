#pragma once
namespace YK {
struct SAlgebraicRegularizationSpec {
    // Borrowed until initialize returns; null/empty/none disables the plugin.
    const char* plugin_id = nullptr;
    // Huber uses internal automatic Prox: inner_iterations/step_size/strength_reduction
    // remain for ABI compatibility and other plugins, not Huber solver control.
    float strength = 0.f, huber_delta = 1e-3f, step_size = 0.f;
    int inner_iterations = 5, dimensions = 3;
    float epsilon = 1e-6f, strength_reduction = 1.f;
    bool non_negative = true;
};
}
