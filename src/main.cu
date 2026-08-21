#include "color.h"
#include "camera.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include "material.h"
#include "bvh.h"
#include "pdf.h"
#include "scenes/window_tree.h"
#include "thunder.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>

D color ray_color(const ray& r, int depth, const color& background,
                  const scene_object* device_objects, int num_objects,
                  const texture_data* device_textures, int num_textures,
                  const material_data* device_materials, int num_materials,
                  bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
                  int* device_prim_indices, int num_prim_indices, int* device_light_indices,
                  int num_lights, curandState* state = nullptr) {
    hit_record rec;
    color attenuation(1.0, 1.0, 1.0);
    color radiance(0, 0, 0);
    color throughput(1, 1, 1);
    ray cur = r;
    while (0 < depth--) {
        if (!hit_bvh(cur, interval(0.001, infinity), rec, device_bvh_nodes, device_prim_indices,
                     device_objects, state)) {
            return radiance + throughput * background;
        }
        const material_data& mat = device_materials[rec.material_id];
        color color_from_emission = emitted(cur, rec, device_textures, mat);
        radiance += throughput * color_from_emission;
        scatter_record srec;
        if (!scatter(cur, rec, srec, device_textures, mat, state)) {
            return radiance;
        }
        if (srec.skip_pdf) {
            throughput *= srec.attenuation;
            cur = srec.skip_pdf_ray;
            continue;
        }
        ray scattered;
        double pdf_val = 0.0;
        scattered =
            ray(rec.p,
                mixture_pdf_generate(rec.p, srec.pdf_axis, device_objects, device_light_indices,
                                     num_lights, srec.pdf_type, srec.anisotropy, state),
                cur.time());
        pdf_val = mixture_pdf_value(rec.p, scattered.direction(), srec.pdf_axis, device_objects,
                                    device_light_indices, num_lights, srec.pdf_type,
                                    srec.anisotropy);

        if (pdf_val <= 0 || std::isnan(pdf_val)) {
            return radiance;
        }

        double spdf = scattering_pdf(cur, rec, scattered, mat);
        throughput *= srec.attenuation * spdf / pdf_val;

        cur = scattered;
    }
    return color(0, 0, 0);
}

__global__ void init_rand_state(curandState* rand_states, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    int pixel_index = (j * width + i);
    curand_init(/*seed=*/19971122, pixel_index, 0, &rand_states[pixel_index]);
}

// CUDA equivalents of the small HLSL helpers used by Unity's Window shader.
D inline float rain_saturate(float value) { return fminf(fmaxf(value, 0.0f), 1.0f); }

D inline float rain_frac(float value) { return value - floorf(value); }

D inline float rain_smoothstep(float edge0, float edge1, float value) {
    float t = rain_saturate((value - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

D inline float rain_n21(float2 p) {
    p.x = rain_frac(p.x * 123.34f);
    p.y = rain_frac(p.y * 345.45f);
    float offset = p.x * (p.x + 34.345f) + p.y * (p.y + 34.345f);
    p.x += offset;
    p.y += offset;
    return rain_frac(p.x * p.y);
}

// Returns (distortion x, distortion y, wet-trail mask).
D inline float3 rain_layer(float2 input_uv, float time, float size) {
    float2 uv = make_float2(input_uv.x * size * 2.0f, input_uv.y * size);
    uv.y += time * 0.25f;

    float2 grid = make_float2(rain_frac(uv.x) - 0.5f, rain_frac(uv.y) - 0.5f);
    float2 id = make_float2(floorf(uv.x), floorf(uv.y));
    float n = rain_n21(id);

    float layer_time = time + n * 2.0f * static_cast<float>(pi);
    float w = input_uv.y * 10.0f;
    float x = (n - 0.5f) * 0.8f;
    x += (0.4f - fabsf(x)) * sinf(3.0f * w) * powf(sinf(w), 6.0f) * 0.45f;

    float y = -sinf(layer_time + sinf(layer_time + sinf(layer_time) * 0.5f)) * 0.45f;
    y -= (grid.x - x) * (grid.x - x);

    // The Unity shader divides by aspect=(2,1).
    float2 drop_pos = make_float2((grid.x - x) * 0.5f, grid.y - y);
    float drop =
        rain_smoothstep(0.05f, 0.03f, hypotf(drop_pos.x, drop_pos.y));

    float2 trail_pos =
        make_float2((grid.x - x) * 0.5f, grid.y - layer_time * 0.25f);
    trail_pos.y = (rain_frac(trail_pos.y * 8.0f) - 0.5f) / 8.0f;
    float trail =
        rain_smoothstep(0.03f, 0.01f, hypotf(trail_pos.x, trail_pos.y));

    float fog_trail = rain_smoothstep(-0.05f, 0.05f, drop_pos.y);
    fog_trail *= rain_smoothstep(0.5f, y, grid.y);
    trail *= fog_trail;
    fog_trail *= rain_smoothstep(0.05f, 0.04f, fabsf(drop_pos.x));

    return make_float3(drop * drop_pos.x + trail * trail_pos.x,
                       drop * drop_pos.y + trail * trail_pos.y, fog_trail);
}

D inline unsigned char rain_read_channel(const unsigned char* image, int width, int height, int x,
                                         int y, int channel) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return image[(y * width + x) * 3 + channel];
}

// Unity's tex2D uses bilinear filtering. The renderer stores rows top-to-bottom,
// whereas Unity screen UVs have their origin at the bottom-left.
D inline float3 rain_sample_bilinear(const unsigned char* image, int width, int height, float2 uv) {
    uv.x = rain_saturate(uv.x);
    uv.y = rain_saturate(uv.y);

    float pixel_x = uv.x * width - 0.5f;
    float pixel_y = (1.0f - uv.y) * height - 0.5f;
    int x0 = static_cast<int>(floorf(pixel_x));
    int y0 = static_cast<int>(floorf(pixel_y));
    float tx = pixel_x - x0;
    float ty = pixel_y - y0;

    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    for (int channel = 0; channel < 3; ++channel) {
        float c00 = rain_read_channel(image, width, height, x0, y0, channel);
        float c10 = rain_read_channel(image, width, height, x0 + 1, y0, channel);
        float c01 = rain_read_channel(image, width, height, x0, y0 + 1, channel);
        float c11 = rain_read_channel(image, width, height, x0 + 1, y0 + 1, channel);
        float top = c00 + (c10 - c00) * tx;
        float bottom = c01 + (c11 - c01) * tx;
        float value = top + (bottom - top) * ty;
        if (channel == 0) result.x = value;
        if (channel == 1) result.y = value;
        if (channel == 2) result.z = value;
    }
    return result;
}

__global__ void rainy_window_kernel(const unsigned char* input_image,
                                    const unsigned char* window_mask,
                                    unsigned char* output_image, int width, int height,
                                    rain_settings rain) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel = y * width + x;
    int pixel_index = pixel * 3;
    if (window_mask[pixel] == 0) {
        output_image[pixel_index] = input_image[pixel_index];
        output_image[pixel_index + 1] = input_image[pixel_index + 1];
        output_image[pixel_index + 2] = input_image[pixel_index + 2];
        return;
    }

    float2 uv = make_float2((x + 0.5f) / width, 1.0f - (y + 0.5f) / height);
    float time = fmodf(rain.time + rain.time_offset, 7200.0f);

    float3 drops = rain_layer(uv, time, rain.size);
    float3 next = rain_layer(make_float2(uv.x * 1.23f + 7.54f, uv.y * 1.23f + 7.54f),
                             time, rain.size);
    drops.x += next.x;
    drops.y += next.y;
    drops.z += next.z;
    next = rain_layer(make_float2(uv.x * 1.35f + 1.54f, uv.y * 1.35f + 1.54f), time,
                      rain.size);
    drops.x += next.x;
    drops.y += next.y;
    drops.z += next.z;
    next = rain_layer(make_float2(uv.x * 1.57f - 7.54f, uv.y * 1.57f - 7.54f), time,
                      rain.size);
    drops.x += next.x;
    drops.y += next.y;
    drops.z += next.z;

    // fwidth(uv.x) is approximately one pixel in normalized screen coordinates.
    float fade = 1.0f - rain_saturate(60.0f / width);
    float2 projected_uv =
        make_float2(uv.x + drops.x * rain.distortion * fade,
                    uv.y + drops.y * rain.distortion * fade);
    float blur = rain.blur * 7.0f * (1.0f - drops.z) * 0.01f;

    constexpr int sample_count = 32;
    float angle = rain_n21(uv) * 2.0f * static_cast<float>(pi);
    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < sample_count; ++sample) {
        float distance =
            sqrtf(rain_frac(sinf((sample + 1) * 546.0f) * 5424.0f));
        float2 offset =
            make_float2(sinf(angle) * blur * distance, cosf(angle) * blur * distance);
        float3 sample_color = rain_sample_bilinear(
            input_image, width, height,
            make_float2(projected_uv.x + offset.x, projected_uv.y + offset.y));
        color_sum.x += sample_color.x;
        color_sum.y += sample_color.y;
        color_sum.z += sample_color.z;
        angle += 1.0f;
    }

    output_image[pixel_index] =
        static_cast<unsigned char>(rain_saturate(color_sum.x / (255.0f * sample_count)) * 255.0f);
    output_image[pixel_index + 1] =
        static_cast<unsigned char>(rain_saturate(color_sum.y / (255.0f * sample_count)) * 255.0f);
    output_image[pixel_index + 2] =
        static_cast<unsigned char>(rain_saturate(color_sum.z / (255.0f * sample_count)) * 255.0f);
}

__global__ void render_kernel(unsigned char* image, int width, int height, camera cam,
                              scene_object* device_objects, int num_objects,
                              texture_data* device_textures, int num_textures,
                              material_data* device_materials, int num_materials,
                              bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
                              int* device_prim_indices, int num_prim_indices,
                              int* device_light_indices, int num_lights,
                              curandState* rand_states = nullptr,
                              unsigned char* window_mask = nullptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    auto pixel_center =
        cam.pixel_sample_start + i * cam.viewport_pixel_u_delta + j * cam.viewport_pixel_v_delta;
    auto ray_direction = pixel_center - cam.origin;
    ray r(cam.origin, ray_direction);
    color pixel_color(0, 0, 0);
    int pixel_index = (j * width + i) * 3;
    curandState local_rand_state = rand_states[pixel_index / 3];
    for (int sample = 0; sample < cam.samples_per_pixel; ++sample) {
        r = cam.get_ray(i, j, sample, &local_rand_state);
        pixel_color +=
            ray_color(r, cam.max_depth, cam.background, device_objects, num_objects,
                      device_textures, num_textures, device_materials, num_materials,
                      device_bvh_nodes, num_bvh_nodes, root_node_index, device_prim_indices,
                      num_prim_indices, device_light_indices, num_lights, &local_rand_state);
    }
    pixel_color /= static_cast<double>(cam.samples_per_pixel);
    write_color(image, pixel_index, pixel_color);

    if (window_mask != nullptr) {
        hit_record primary_rec;
        ray primary_ray(cam.origin, pixel_center - cam.origin);
        bool primary_hit =
            hit_bvh(primary_ray, interval(0.001, infinity), primary_rec, device_bvh_nodes,
                    device_prim_indices, device_objects, &local_rand_state);
        window_mask[pixel_index / 3] =
            primary_hit &&
                    device_materials[primary_rec.material_id].receives_rain_post_process
                ? 1
                : 0;
    }
}

void camera::render(int width, int height, scene_object* host_objects, int num_objects,
                    texture_data* host_textures, int num_textures, material_data* host_materials,
                    int num_materials, bvh_node* host_bvh_nodes, int num_bvh_nodes,
                    int root_node_index, int* prim_indices, int num_prim_indices,
                    int* light_indices, int num_lights, const rain_settings& rain, int frame) {
    assert(width == this->image_width && height == this->image_height &&
           "Image dimensions must match camera dimensions");
    dim3 block_size = dim3(16, 16);
    dim3 grid_size =
        dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Initialize the random state for each pixel
    curandState* rand_states;
    cudaMalloc(&rand_states, width * height * sizeof(curandState));
    init_rand_state<<<grid_size, block_size>>>(rand_states, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Device memory allocation
    unsigned char* image;
    size_t image_size = width * height * 3;
    cudaMalloc(&image, image_size);
    texture_data* device_textures;
    cudaMalloc(&device_textures, num_textures * sizeof(texture_data));
    cudaMemcpy(device_textures, host_textures, num_textures * sizeof(texture_data),
               cudaMemcpyHostToDevice);
    material_data* device_materials;
    cudaMalloc(&device_materials, num_materials * sizeof(material_data));
    cudaMemcpy(device_materials, host_materials, num_materials * sizeof(material_data),
               cudaMemcpyHostToDevice);
    scene_object* device_objects;
    cudaMalloc(&device_objects, num_objects * sizeof(scene_object));
    cudaMemcpy(device_objects, host_objects, num_objects * sizeof(scene_object),
               cudaMemcpyHostToDevice);
    int* device_prim_indices;
    cudaMalloc(&device_prim_indices, num_prim_indices * sizeof(int));
    cudaMemcpy(device_prim_indices, prim_indices, num_prim_indices * sizeof(int),
               cudaMemcpyHostToDevice);
    bvh_node* device_bvh_nodes;
    cudaMalloc(&device_bvh_nodes, num_bvh_nodes * sizeof(bvh_node));
    cudaMemcpy(device_bvh_nodes, host_bvh_nodes, num_bvh_nodes * sizeof(bvh_node),
               cudaMemcpyHostToDevice);
    int* device_light_indices = nullptr;
    if (num_lights > 0) {
        assert(light_indices != nullptr && "Light indices must not be empty.");
        cudaMalloc(&device_light_indices, num_lights * sizeof(int));
        cudaMemcpy(device_light_indices, light_indices, num_lights * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    bool use_post_processing_rain =
        rain.mode == rain_mode::POST_PROCESSING || rain.mode == rain_mode::HYBRID;
    unsigned char* window_mask = nullptr;
    unsigned char* rainy_image = nullptr;
    if (use_post_processing_rain) {
        cudaMalloc(&window_mask, width * height * sizeof(unsigned char));
        cudaMalloc(&rainy_image, image_size);
    }

    render_kernel<<<grid_size, block_size>>>(
        image, width, height, *this, device_objects, num_objects, device_textures, num_textures,
        device_materials, num_materials, device_bvh_nodes, num_bvh_nodes, root_node_index,
        device_prim_indices, num_prim_indices, device_light_indices, num_lights, rand_states,
        window_mask);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> host_image(image_size);
    std::filesystem::create_directories("build/frames");

    auto write_frame = [&](const unsigned char* device_image, int output_frame) {
        CUDA_CHECK(
            cudaMemcpy(host_image.data(), device_image, image_size, cudaMemcpyDeviceToHost));
        std::ostringstream name;
        name << "build/frames/frame_" << std::setw(4) << std::setfill('0') << output_frame
             << ".ppm";
        std::ofstream out(name.str(), std::ios::binary);
        write_image(out, host_image, width, height);
        std::clog << "\rframe " << output_frame << " done" << std::flush;
    };

    if (use_post_processing_rain) {
        int frame_count = max(1, rain.frame_count);
        float fps = fmaxf(rain.frames_per_second, 0.001f);
        for (int animation_frame = 0; animation_frame < frame_count; ++animation_frame) {
            rain_settings animated_rain = rain;
            animated_rain.time =
                rain.time + animation_frame * rain.time_scale / fps;
            rainy_window_kernel<<<grid_size, block_size>>>(image, window_mask, rainy_image, width,
                                                           height, animated_rain);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            write_frame(rainy_image, frame + animation_frame);
        }
    } else {
        write_frame(image, frame);
    }

    // Free the device memory
    cudaFree(device_objects);
    cudaFree(device_textures);
    cudaFree(device_materials);
    cudaFree(device_bvh_nodes);
    cudaFree(device_prim_indices);
    cudaFree(device_light_indices);
    cudaFree(window_mask);
    cudaFree(rainy_image);
    cudaFree(image);
    cudaFree(rand_states);
}

int main(int argc, char** argv) {
    rain_settings rain;
    cloud_settings clouds;
    lightning_settings lightning;
    render_quality_settings quality;
    rain.mode = rain_mode::POST_PROCESSING;
    rain.size = 3.7f;
    rain.distortion = -5.0f;
    rain.blur = 0.05f;

    int start_frame = 0;
    auto print_usage = [&]() {
        std::cout
            << "Usage: " << argv[0]
            << " [--fast] [--width N] [--samples N] [--max-depth N]"
               " [--frames N] [--fps N] [--start-frame N] [--time-scale N]"
               " [--clouds] [--cloud-speed N] [--cloud-density N]"
               " [--cloud-start-x N] [--cloud-count N]"
               " [--lightning] [--lightning-first N]"
               " [--lightning-interval N] [--lightning-intensity N]"
               " [--lightning-seed N] [--thunder-delay N] [--thunder-volume N]"
               " [--no-thunder] [--no-rain]\n";
    };

    try {
        for (int arg = 1; arg < argc; ++arg) {
            std::string option = argv[arg];
            if (option == "--help" || option == "-h") {
                print_usage();
                return 0;
            }
            if (option == "--clouds") {
                clouds.enabled = true;
                continue;
            }
            if (option == "--fast") {
                quality.image_width = 640;
                quality.samples_per_pixel = 9;
                quality.max_depth = 6;
                continue;
            }
            if (option == "--no-rain") {
                rain.mode = rain_mode::NONE;
                continue;
            }
            if (option == "--lightning") {
                lightning.enabled = true;
                continue;
            }
            if (option == "--no-thunder") {
                lightning.thunder_enabled = false;
                continue;
            }
            if (arg + 1 >= argc) {
                throw std::invalid_argument("missing value for " + option);
            }
            std::string value = argv[++arg];
            if (option == "--width") {
                quality.image_width = std::stoi(value);
            } else if (option == "--samples") {
                quality.samples_per_pixel = std::stoi(value);
            } else if (option == "--max-depth") {
                quality.max_depth = std::stoi(value);
            } else if (option == "--frames") {
                rain.frame_count = std::stoi(value);
            } else if (option == "--fps") {
                rain.frames_per_second = std::stof(value);
            } else if (option == "--start-frame") {
                start_frame = std::stoi(value);
            } else if (option == "--time-scale") {
                rain.time_scale = std::stof(value);
            } else if (option == "--cloud-speed") {
                clouds.enabled = true;
                clouds.speed = std::stod(value);
            } else if (option == "--cloud-density") {
                clouds.enabled = true;
                clouds.density = std::stod(value);
            } else if (option == "--cloud-start-x") {
                clouds.enabled = true;
                clouds.start_x = std::stod(value);
            } else if (option == "--cloud-count") {
                clouds.enabled = true;
                clouds.bank_count = std::stoi(value);
            } else if (option == "--lightning-first") {
                lightning.enabled = true;
                lightning.first_strike = std::stod(value);
            } else if (option == "--lightning-interval") {
                lightning.enabled = true;
                lightning.interval = std::stod(value);
            } else if (option == "--lightning-intensity") {
                lightning.enabled = true;
                lightning.intensity = std::stod(value);
            } else if (option == "--lightning-seed") {
                lightning.enabled = true;
                lightning.seed = static_cast<std::uint32_t>(std::stoul(value));
            } else if (option == "--thunder-delay") {
                lightning.enabled = true;
                lightning.thunder_delay = std::stod(value);
            } else if (option == "--thunder-volume") {
                lightning.enabled = true;
                lightning.thunder_volume = std::stod(value);
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        print_usage();
        return 1;
    }

    if (quality.image_width < 16 || quality.samples_per_pixel < 1 || quality.max_depth < 1 ||
        rain.frame_count < 1 || rain.frames_per_second <= 0.0f || start_frame < 0 ||
        clouds.bank_count < 1 ||
        clouds.speed < 0.0 || clouds.density <= 0.0 || lightning.first_strike < 0.0 ||
        lightning.interval <= 0.0 || lightning.intensity <= 0.0 ||
        lightning.thunder_delay < 0.0 || lightning.thunder_volume < 0.0) {
        std::cerr << "Width must be at least 16. Samples, depth, frames, FPS, cloud count, and "
                     "cloud density must be positive; start frame and cloud speed cannot be "
                     "negative. Lightning timing, intensity, and thunder settings must also "
                     "be valid.\n";
        return 1;
    }

    std::clog << "render settings: " << quality.image_width << "px wide, "
              << quality.samples_per_pixel << " sample(s)/pixel, depth " << quality.max_depth
              << ", " << clouds.bank_count << " cloud bank(s)" << std::endl;

    int requested_frames = rain.frame_count;
    bool dynamic_scene = clouds.enabled || lightning.enabled;
    if (dynamic_scene) {
        // Moving clouds and lightning change geometry or illumination, so each
        // frame needs a new BVH and path-traced base image. Rain remains a
        // post-process applied once to each newly rendered frame.
        rain.frame_count = 1;
        for (int animation_frame = 0; animation_frame < requested_frames; ++animation_frame) {
            int output_frame = start_frame + animation_frame;
            double scene_time = output_frame / rain.frames_per_second;
            rain.time = static_cast<float>(scene_time * rain.time_scale);
            render_window_tree_scene(scene_time, output_frame, rain, clouds, lightning, quality);
        }
    } else {
        rain.time = start_frame * rain.time_scale / rain.frames_per_second;
        render_window_tree_scene(rain.time, start_frame, rain, clouds, lightning, quality);
    }

    if (lightning.enabled && lightning.thunder_enabled) {
        double video_start_time = start_frame / rain.frames_per_second;
        double video_duration = requested_frames / rain.frames_per_second;
        int thunder_events = write_thunder_wav("build/thunder.wav", video_start_time,
                                               video_duration, lightning, clouds);
        if (thunder_events >= 0) {
            std::clog << "generated build/thunder.wav with " << thunder_events
                      << " audible thunder event(s)" << std::endl;
        } else {
            std::cerr << "Could not write build/thunder.wav\n";
        }
    }
    std::clog << std::endl;
    return 0;
}
