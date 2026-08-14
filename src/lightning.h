#pragma once

#include "cloud.h"
#include "vec3.h"

#include <cmath>
#include <cstdint>

struct lightning_settings {
    bool enabled = false;
    bool thunder_enabled = true;
    double first_strike = 0.8;
    double interval = 6.0;
    double intensity = 450.0;
    double sky_flash = 3.0;
    double thunder_delay = 1.0;
    double thunder_volume = 0.85;
    std::uint32_t seed = 19971122u;
};

struct lightning_event {
    int index = -1;
    double time = 0.0;
    point3 cloud_point;
    point3 ground_point;
    int screen_region = 1;       // 0 left, 1 middle, 2 right.
    bool behind_mountains = false;
    double visual_scale = 1.0;   // Keeps distant bolts readable in perspective.
};

struct lightning_frame_state {
    bool active = false;
    double flash = 0.0;
    lightning_event event;
};

inline std::uint32_t lightning_hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

inline double lightning_random01(std::uint32_t seed, std::uint32_t stream) {
    return lightning_hash(seed ^ lightning_hash(stream + 0x9e3779b9u)) /
           static_cast<double>(0xffffffffu);
}

inline double lightning_strike_time(int index, const lightning_settings& settings) {
    if (index == 0) return settings.first_strike;
    double jitter = (lightning_random01(settings.seed, static_cast<std::uint32_t>(index)) - 0.5) *
                    settings.interval * 0.30;
    return settings.first_strike + index * settings.interval + jitter;
}

inline lightning_event make_lightning_event(int index, const lightning_settings& settings,
                                             const cloud_settings&) {
    lightning_event event;
    event.index = index;
    event.time = lightning_strike_time(index, settings);

    double r0 = lightning_random01(settings.seed, 1000u + index * 7u);
    double r3 = lightning_random01(settings.seed, 1003u + index * 7u);
    double r4 = lightning_random01(settings.seed, 1004u + index * 7u);
    double r5 = lightning_random01(settings.seed, 1005u + index * 7u);
    double r6 = lightning_random01(settings.seed, 1006u + index * 7u);

    // Choose left, middle, or right in a shuffled order. World +X appears on
    // screen-left for this camera, hence the reversed centers below.
    // Shuffle the three regions independently in each group of three events.
    // This retains a random order while preventing a long video from, by bad
    // luck, placing every visible strike in the same part of the sky.
    int region_order[] = {0, 1, 2};
    int region_block = index / 3;
    for (int position = 2; position > 0; --position) {
        double shuffle_random = lightning_random01(
            settings.seed, 8000u + static_cast<std::uint32_t>(region_block * 3 + position));
        int swap_position = static_cast<int>(shuffle_random * (position + 1));
        if (swap_position > position) swap_position = position;
        int saved = region_order[position];
        region_order[position] = region_order[swap_position];
        region_order[swap_position] = saved;
    }
    event.screen_region = region_order[index % 3];

    // Randomize the near/far order in pairs, guaranteeing that both depth
    // layers remain represented even in a relatively short animation.
    int depth_block = index / 2;
    bool first_in_pair_is_far =
        lightning_random01(settings.seed, 9000u + depth_block) < 0.5;
    event.behind_mountains = (index % 2 == 0) ? first_in_pair_is_far : !first_in_pair_is_far;
    event.visual_scale = event.behind_mountains ? 1.75 : 1.0;

    const double region_centers[] = {0.72, 0.0, -0.72};
    double horizontal_extent = event.behind_mountains ? 18.0 : 10.0;
    double region_jitter = (r0 - 0.5) * horizontal_extent * 0.34;
    double strike_x = region_centers[event.screen_region] * horizontal_extent + region_jitter;

    // Far strikes sit behind the mountain range, so its geometry naturally
    // occludes their lower sections. Near strikes remain in front of the
    // mountains but behind the garden, fence, and trees.
    double strike_z = event.behind_mountains ? 58.0 + 7.0 * r3 : 25.0 + 9.0 * r3;
    double strike_y = event.behind_mountains ? 15.0 + 3.0 * r4 : 12.0 + 2.5 * r4;
    event.cloud_point = point3(strike_x, strike_y, strike_z);
    event.ground_point =
        point3(strike_x + (r5 - 0.5) * 5.0, 0.05, strike_z + (r6 - 0.5) * 3.0);
    return event;
}

inline double lightning_flash_envelope(double age) {
    if (age < 0.0 || age > 0.35) return 0.0;

    double first = std::exp(-age * 24.0);
    double second = age >= 0.085 ? 0.85 * std::exp(-(age - 0.085) * 30.0) : 0.0;
    double third = age >= 0.19 ? 0.45 * std::exp(-(age - 0.19) * 34.0) : 0.0;
    return std::fmin(1.0, first + second + third);
}

inline lightning_frame_state lightning_at_time(double time, const lightning_settings& settings,
                                                const cloud_settings& clouds) {
    lightning_frame_state state;
    if (!settings.enabled || time < 0.0) return state;

    // This covers more than 50 minutes with the default interval while keeping
    // event lookup deterministic and independent of render order.
    for (int index = 0; index < 512; ++index) {
        double strike_time = lightning_strike_time(index, settings);
        if (strike_time > time + 0.35) break;
        double flash = lightning_flash_envelope(time - strike_time);
        if (flash > state.flash) {
            state.active = true;
            state.flash = flash;
            state.event = make_lightning_event(index, settings, clouds);
        }
    }
    return state;
}
