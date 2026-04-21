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
        if(hit(r, interval(0, closest_so_far), rec, device_objects[i])){
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

void camera::render(unsigned char* image, int width, int height, scene_object* device_objects, int num_objects){
    assert(width == this->image_width && height == this->image_height && "Image dimensions must match camera dimensions");
    dim3 block_size = dim3(16, 16);
    dim3 grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    render_kernel<<<grid_size, block_size>>>(image, width, height, *this, device_objects, num_objects);
}

int main(){
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;

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
    camera cam;
    cam.init(image_width, aspect_ratio, focal_length, viewport_height, point3(0, 0, 0));
    int image_height = cam.image_height;

    // Device memory allocation
    unsigned char* image;
    size_t image_size = image_width * image_height * 3;
    cudaMalloc(&image, image_size);

    scene_object* device_objects;
    cudaMalloc(&device_objects, num_objects * sizeof(scene_object));
    cudaMemcpy(device_objects, host_objects, num_objects * sizeof(scene_object), cudaMemcpyHostToDevice);
    
    cam.render(image, image_width, image_height, device_objects, num_objects);

    cudaDeviceSynchronize();
    std::vector<unsigned char> host_image(image_size);
    cudaMemcpy(host_image.data(), image, image_size, cudaMemcpyDeviceToHost);
    cudaFree(image);
    write_image(std::cout, host_image, image_width, image_height);
    return 0;
}