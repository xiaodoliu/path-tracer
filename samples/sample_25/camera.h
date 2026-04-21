# pragma once

#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "bvh.h"
#include <cuda_runtime.h>

struct scene_object;

struct camera{
    point3 origin = point3(0, 0, 0);
    double focal_length = 1.0;
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    int image_height = 450;
    int samples_per_pixel = 100;
    int max_depth = 50;
    double vfov = 90.0;
    point3 lookfrom = point3(0, 0, 0);
    point3 lookat = point3(0, 0, -1);
    vec3 vup = vec3(0, 1, 0);

    double defocus_angle = 0.0;
    double focus_dist = 10.0;

    point3 pixel_sample_start;
    vec3 viewport_pixel_u_delta;
    vec3 viewport_pixel_v_delta;
    vec3 u, v, w;
    vec3 defocus_disk_u, defocus_disk_v;

    void init(int image_width, int samples_per_pixel, int max_depth,
        double aspect_ratio, double vfov, 
        point3 lookfrom, point3 lookat, vec3 vup, double defocus_angle, double focus_dist){
        this->lookfrom = lookfrom;
        this->lookat = lookat;
        this->vup = vup;
        this->defocus_angle = defocus_angle;
        this->focus_dist = focus_dist;
        this->origin = lookfrom;
        this->focal_length = focus_dist;
        this->vfov = vfov;
        this->image_width = image_width;
        this->samples_per_pixel = samples_per_pixel;
        this->max_depth = max_depth;
        this->aspect_ratio = aspect_ratio;
        int image_height = static_cast<int>(image_width / aspect_ratio);
        this->image_height = (image_height < 1) ? 1 : image_height;
        double theta = degrees_to_radians(vfov);
        double viewport_height = 2 * focal_length * std::tan(theta / 2);
        double viewport_width = viewport_height * static_cast<double>(image_width) / image_height;
        w = normalize(lookfrom-lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);
        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = -viewport_height * v;
        this->viewport_pixel_u_delta = viewport_u / image_width;
        this->viewport_pixel_v_delta = viewport_v / image_height;
        auto viewport_upper_left = this->origin - focal_length * w - viewport_u / 2 - viewport_v / 2;
        this->pixel_sample_start  = viewport_upper_left + this->viewport_pixel_u_delta / 2 + this->viewport_pixel_v_delta / 2;

        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = defocus_radius * u;
        defocus_disk_v = defocus_radius * v;
    }

    D point3 defocus_disk_sample(curandState* state){
        auto random_disk_vector = random_in_unit_disk(state);
        return origin + random_disk_vector.x() * defocus_disk_u + random_disk_vector.y() * defocus_disk_v;
    }

    D ray get_ray(int i, int j, curandState* state){
        auto pixel_sample = pixel_sample_start + 
            (i+(random_double(state)-0.5)) * viewport_pixel_u_delta + 
            (j+(random_double(state)-0.5)) * viewport_pixel_v_delta;
        auto ray_origin = defocus_angle > 0 ? defocus_disk_sample(state) : origin;
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_double(state);
        return ray(ray_origin, ray_direction, ray_time);
    }

    void render(int width, int height,
        scene_object* host_objects, int num_objects,
        texture_data* host_textures, int num_textures,
        material_data* host_materials, int num_materials,
        bvh_node* host_bvh_nodes, int num_bvh_nodes, int root_node_index,
        int* prim_indices, int num_prim_indices);
        
};
