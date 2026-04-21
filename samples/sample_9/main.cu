#include "color.h"
#include "camera.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

D color ray_color(const ray& r, int depth, const scene_object* device_objects, int num_objects, curandState* state){
    hit_record rec;
    double bouncing_scale = 1.0;
    ray cur = r;
    while(0 < depth--){
        bool hit_anything = false;
        double closest_so_far = infinity;
        for(int i = 0; i < num_objects; ++i){
            if(hit(cur, interval(0.001, closest_so_far), rec, device_objects[i])){
                hit_anything = true;
                closest_so_far = rec.t;
            }
        }
        if(hit_anything){
            bouncing_scale *= 0.5;
            vec3 direction = random_on_hemisphere(rec.normal, state);
            cur = ray(rec.p, direction);
        }else{
            vec3 unit_direction = normalize(cur.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            return bouncing_scale * ((1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0));
        }
    }
    return color(0, 0, 0);
}

__global__ void init_rand_state(curandState* rand_states, int width, int height){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    int pixel_index = (j * width + i);
    curand_init(/*seed=*/19260817, pixel_index, 0, &rand_states[pixel_index]);
}

__global__ void render_kernel(unsigned char* image, int width, int height, camera cam, scene_object* device_objects, int num_objects, curandState* rand_states = nullptr){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    auto pixel_center = cam.pixel_sample_start + i * cam.viewport_pixel_u_delta + j * cam.viewport_pixel_v_delta;
    auto ray_direction = pixel_center - cam.origin;
    ray r(cam.origin, ray_direction);
    color pixel_color(0, 0, 0);
    int pixel_index = (j * width + i) * 3;
    curandState local_rand_state = rand_states[pixel_index/3];
    for(int s = 0; s < cam.samples_per_pixel; ++s){
        r = cam.get_ray(i, j, &local_rand_state);
        pixel_color += ray_color(r, cam.max_depth, device_objects, num_objects, &local_rand_state);
    }  
    pixel_color /= (cam.samples_per_pixel * 1.0);   
    // if(rand_states == nullptr){
    //     pixel_color = ray_color(r, device_objects, num_objects);
    // }else{
    //     curandState local_rand_state = rand_states[pixel_index/3];
    //     for(int s = 0; s < cam.samples_per_pixel; ++s){
    //         r = cam.get_ray(i, j, &local_rand_state);
    //         pixel_color += ray_color(r, device_objects, num_objects);
    //     }  
    //     pixel_color /= (cam.samples_per_pixel * 1.0);  
    // }
    write_color(image, pixel_index, pixel_color);
}

void camera::render(int width, int height, scene_object* host_objects, int num_objects){
    assert(width == this->image_width && height == this->image_height && "Image dimensions must match camera dimensions");
    dim3 block_size = dim3(16, 16);
    dim3 grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Initialize the random state for each pixel
    curandState* rand_states;
    cudaMalloc(&rand_states, width * height * sizeof(curandState));
    init_rand_state<<<grid_size, block_size>>>(rand_states, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Device memory allocation
    unsigned char* image;
    size_t image_size = width * height * 3;
    cudaMalloc(&image, image_size);
    scene_object* device_objects;
    cudaMalloc(&device_objects, num_objects * sizeof(scene_object));
    cudaMemcpy(device_objects, host_objects, num_objects * sizeof(scene_object), cudaMemcpyHostToDevice);

    render_kernel<<<grid_size, block_size>>>(image, width, height, *this, device_objects, num_objects, rand_states);
    CUDA_CHECK(cudaGetLastError());

    // Copy the image back to the host
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> host_image(image_size);
    cudaMemcpy(host_image.data(), image, image_size, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(device_objects);
    cudaFree(image);
    cudaFree(rand_states);
    write_image(std::cout, host_image, width, height);
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
    cam.init(image_width, /*samples_per_pixel=*/100, /*max_depth=*/50, aspect_ratio, focal_length, viewport_height, point3(0, 0, 0));
    int image_height = cam.image_height;
    
    cam.render(image_width, image_height, host_objects, num_objects);

    return 0;
}