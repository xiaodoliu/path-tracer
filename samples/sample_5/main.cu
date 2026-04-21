#include "color.h"
#include "camera.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

D color ray_color(const ray& r, const scene_object* device_objects, int num_objects){
    hit_record rec;
    bool hit_anything = false;
    double closest_so_far = infinity;
    for(int i = 0; i < num_objects; ++i){
        if(hit(r, 0, closest_so_far, rec, device_objects[i])){
            hit_anything = true;
            closest_so_far = rec.t;
        }
    }
    if(hit_anything){
        return 0.5 * color(rec.normal + point3(1, 1, 1));
    }
    vec3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render_kernel(unsigned char* image, int width, int height, camera cam, scene_object* device_objects, int num_objects){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    auto pixel_center = cam.pixel_sample_start + i * cam.viewport_pixel_u_delta + j * cam.viewport_pixel_v_delta;
    auto ray_direction = pixel_center - cam.origin;
    ray r(cam.origin, ray_direction);
    color pixel_color = ray_color(r, device_objects, num_objects);
    int pixel_index = (j * width + i) * 3;
    write_color(image, pixel_index, pixel_color);
}


int main(){
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    //world
    const int num_objects = 2;
    scene_object host_objects[num_objects];
    host_objects[0].type = object_type::SPHERE;
    host_objects[0].sphere_data = sphere(point3(0, 0, -1), 0.5);
    host_objects[1].type = object_type::SPHERE;
    host_objects[1].sphere_data = sphere(point3(0, -100.5, -1), 100);

    //camera
    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * static_cast<double>(image_width) / image_height;
    vec3 viewport_u(viewport_width, 0, 0);
    vec3 viewport_v(0, -viewport_height, 0);
    vec3 viewport_pixel_u_delta = viewport_u / image_width;
    vec3 viewport_pixel_v_delta = viewport_v / image_height;
    camera cam;
    cam.origin = point3(0, 0, 0);
    cam.focal_length = focal_length;
    cam.viewport_pixel_u_delta = viewport_pixel_u_delta;
    cam.viewport_pixel_v_delta = viewport_pixel_v_delta;
    auto viewport_upper_left = cam.origin - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    cam.pixel_sample_start  = viewport_upper_left + viewport_pixel_u_delta / 2 + viewport_pixel_v_delta / 2;

    // Device memory allocation
    unsigned char* image;
    size_t image_size = image_width * image_height * 3;
    cudaMalloc(&image, image_size);

    scene_object* device_objects;
    cudaMalloc(&device_objects, num_objects * sizeof(scene_object));
    cudaMemcpy(device_objects, host_objects, num_objects * sizeof(scene_object), cudaMemcpyHostToDevice);
    
    dim3 block_size = dim3(16, 16);
    dim3 grid_size = dim3((image_width + block_size.x - 1) / block_size.x, (image_height + block_size.y - 1) / block_size.y);
    render_kernel<<<grid_size, block_size>>>(image, image_width, image_height, cam, device_objects, num_objects);
    cudaDeviceSynchronize();
    std::vector<unsigned char> host_image(image_size);
    cudaMemcpy(host_image.data(), image, image_size, cudaMemcpyDeviceToHost);
    cudaFree(image);
    write_image(std::cout, host_image, image_width, image_height);
    return 0;
}