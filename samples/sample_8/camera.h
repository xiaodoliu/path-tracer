# pragma once

#include "vec3.h"
#include "ray.h"
#include <cuda_runtime.h>

struct scene_object;

struct camera{
    point3 origin = point3(0, 0, 0);
    double focal_length = 1.0;
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    int image_height = 450;
    int samples_per_pixel = 10;

    point3 pixel_sample_start;
    vec3 viewport_pixel_u_delta;
    vec3 viewport_pixel_v_delta;

    void init(int image_width, int samples_per_pixel,
        double aspect_ratio, double focal_length, double viewport_height, 
        point3 origin){
        this->origin = origin;
        this->focal_length = focal_length;
        this->image_width = image_width;
        this->samples_per_pixel = samples_per_pixel;
        this->aspect_ratio = aspect_ratio;
        int image_height = static_cast<int>(image_width / aspect_ratio);
        this->image_height = (image_height < 1) ? 1 : image_height;

        double viewport_width = viewport_height * static_cast<double>(image_width) / image_height;
        vec3 viewport_u(viewport_width, 0, 0);
        vec3 viewport_v(0, -viewport_height, 0);
        this->viewport_pixel_u_delta = viewport_u / image_width;
        this->viewport_pixel_v_delta = viewport_v / image_height;
        auto viewport_upper_left = this->origin - vec3(0, 0, this->focal_length) - viewport_u / 2 - viewport_v / 2;
        this->pixel_sample_start  = viewport_upper_left + this->viewport_pixel_u_delta / 2 + this->viewport_pixel_v_delta / 2;
    }

    D ray get_ray(int i, int j, curandState* state){
        auto pixel_sample = pixel_sample_start + 
            (i+(random_double(state)-0.5)) * viewport_pixel_u_delta + 
            (j+(random_double(state)-0.5)) * viewport_pixel_v_delta;
        auto ray_direction = pixel_sample - origin;
        return ray(origin, ray_direction);
    }

    void render(int width, int height,
        scene_object* host_objects, int num_objects);
};