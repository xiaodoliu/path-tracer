#pragma once

#include "lightning.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

inline void thunder_write_u16(std::ofstream& out, std::uint16_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
}

inline void thunder_write_u32(std::ofstream& out, std::uint32_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
    out.put(static_cast<char>((value >> 16) & 0xff));
    out.put(static_cast<char>((value >> 24) & 0xff));
}

inline int write_thunder_wav(const std::string& filename, double video_start_time,
                             double video_duration, const lightning_settings& lightning,
                             const cloud_settings& clouds) {
    constexpr int sample_rate = 48000;
    constexpr double thunder_length = 4.5;
    int sample_count = std::max(1, static_cast<int>(std::ceil(video_duration * sample_rate)));
    std::vector<float> samples(sample_count, 0.0f);
    int audible_events = 0;

    double video_end_time = video_start_time + video_duration;
    for (int event_index = 0; event_index < 512; ++event_index) {
        lightning_event event = make_lightning_event(event_index, lightning, clouds);
        double arrival_time = event.time + lightning.thunder_delay;
        if (arrival_time > video_end_time) break;
        if (arrival_time + thunder_length < video_start_time) continue;
        ++audible_events;

        int event_start_sample =
            static_cast<int>((arrival_time - video_start_time) * sample_rate);
        int first_sample = std::max(0, event_start_sample);
        int last_sample = std::min(
            sample_count, event_start_sample + static_cast<int>(thunder_length * sample_rate));
        if (last_sample <= first_sample) continue;
        std::uint32_t noise_state =
            lightning_hash(lightning.seed ^ static_cast<std::uint32_t>(event_index * 7919 + 17));
        double low = 0.0;
        double sub = 0.0;

        for (int sample = first_sample; sample < last_sample; ++sample) {
            double local_time = (sample - event_start_sample) / static_cast<double>(sample_rate);
            noise_state = noise_state * 1664525u + 1013904223u;
            double white = ((noise_state >> 8) / static_cast<double>(0x00ffffffu)) * 2.0 - 1.0;
            low = 0.965 * low + 0.035 * white;
            sub = 0.995 * sub + 0.005 * low;

            double attack = std::fmin(1.0, local_time * 18.0);
            double decay = std::exp(-local_time * 0.62);
            double rolling = 0.65 + 0.20 * std::sin(local_time * 16.0) +
                             0.15 * std::sin(local_time * 7.3 + 1.7);
            double initial_crack = white * std::exp(-local_time * 13.0);
            double second_roll = local_time >= 0.55
                                     ? std::exp(-(local_time - 0.55) * 1.3) * 0.35
                                     : 0.0;
            double third_roll = local_time >= 1.35
                                    ? std::exp(-(local_time - 1.35) * 1.1) * 0.22
                                    : 0.0;
            double rumble = attack * decay * rolling * (1.8 * sub + 0.65 * low) +
                            0.28 * initial_crack + (second_roll + third_roll) * low;
            samples[sample] += static_cast<float>(lightning.thunder_volume * rumble);
        }
    }

    float peak = 0.0f;
    for (float sample : samples) peak = std::max(peak, std::fabs(sample));
    float normalization = peak > 0.95f ? 0.95f / peak : 1.0f;

    std::filesystem::path path(filename);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(filename, std::ios::binary);
    if (!out) return -1;

    constexpr std::uint16_t channel_count = 1;
    constexpr std::uint16_t bits_per_sample = 16;
    constexpr std::uint16_t block_align = channel_count * bits_per_sample / 8;
    constexpr std::uint32_t byte_rate = sample_rate * block_align;
    std::uint32_t data_size = sample_count * block_align;

    out.write("RIFF", 4);
    thunder_write_u32(out, 36u + data_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    thunder_write_u32(out, 16u);
    thunder_write_u16(out, 1u);  // PCM
    thunder_write_u16(out, channel_count);
    thunder_write_u32(out, sample_rate);
    thunder_write_u32(out, byte_rate);
    thunder_write_u16(out, block_align);
    thunder_write_u16(out, bits_per_sample);
    out.write("data", 4);
    thunder_write_u32(out, data_size);

    for (float sample : samples) {
        float clamped = std::fmax(-1.0f, std::fmin(1.0f, sample * normalization));
        std::int16_t pcm = static_cast<std::int16_t>(clamped * 32767.0f);
        thunder_write_u16(out, static_cast<std::uint16_t>(pcm));
    }
    return audible_events;
}
