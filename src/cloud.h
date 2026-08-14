#pragma once

#include <cstdint>

struct cloud_settings {
    bool enabled = false;
    double speed = 1.5;        // World +X units/second; screen-right to left for this camera.
    double density = 0.45;     // Peak density of each heterogeneous cloud field.
    double start_x = -27.0;    // Center position where recycled clouds enter screen-right.
    int bank_count = 5;        // Banks distributed across the sky at all times.
    double travel_span = 64.0; // Distance before a bank recycles to screen-right.
    std::uint32_t seed = 0x43a6f29du;
};

inline std::uint32_t cloud_hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

inline double cloud_random01(std::uint32_t seed, std::uint32_t stream) {
    return cloud_hash(seed ^ cloud_hash(stream + 0x9e3779b9u)) /
           static_cast<double>(0xffffffffu);
}
