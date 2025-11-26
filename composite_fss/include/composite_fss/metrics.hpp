#pragma once

#include <cstddef>

namespace cfss {

struct GateMetrics {
    std::size_t key_bytes = 0;
    std::size_t lut_bytes = 0;
    std::size_t triples_used = 0;
    double keygen_ms = 0.0;
    double online_ms = 0.0;
};

struct MetricsRegistry {
    GateMetrics gelu;
    GateMetrics silu;
    GateMetrics softmax;
    GateMetrics recip;
    GateMetrics inv;
    GateMetrics norm;
    GateMetrics fused_layer;
};

inline MetricsRegistry &global_metrics() {
    static MetricsRegistry registry;
    return registry;
}

} // namespace cfss

