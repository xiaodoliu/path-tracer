#include "color.h"
#include "camera.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include "material.h"
#include "bvh.h"
#include "pdf.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include <fstream>
#include <sstream>
#include <iomanip>

D color ray_color(const ray& r, int depth, 
    const color& background,
    const scene_object* device_objects, int num_objects, 
    const texture_data* device_textures, int num_textures,
    const material_data* device_materials, int num_materials, 
    bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
    int* device_prim_indices, int num_prim_indices,
    int* device_light_indices, int num_lights,
    curandState* state = nullptr){
    hit_record rec;
    color attenuation(1.0, 1.0, 1.0);
    color radiance(0, 0, 0);
    color throughput(1, 1, 1);
    ray cur = r;
    while(0 < depth--){
        if(!hit_bvh(cur, interval(0.001, infinity), rec, device_bvh_nodes, device_prim_indices, device_objects, state)){
            return radiance + throughput * background;
        }
        const material_data& mat = device_materials[rec.material_id];
        color color_from_emission = emitted(cur, rec, device_textures, mat);
        radiance += throughput * color_from_emission;
        scatter_record srec;
        if(!scatter(cur, rec, srec, device_textures, mat, state)){
            return radiance;
        }
        if(srec.skip_pdf){
            throughput *= srec.attenuation;
            cur = srec.skip_pdf_ray;
            continue;
        }
        ray scattered;
        double pdf_val = 0.0;
        scattered = ray(rec.p, 
            mixture_pdf_generate(
                rec.p, rec.normal, device_objects, device_light_indices, num_lights, 
                srec.pdf_type, state), 
            cur.time());
        pdf_val = mixture_pdf_value(rec.p, scattered.direction(), rec.normal, 
            device_objects, device_light_indices, num_lights, srec.pdf_type);
        
        if(pdf_val <= 0 || std::isnan(pdf_val)) {
            return radiance;
        }
        
        double spdf = scattering_pdf(cur, rec, scattered, mat);
        throughput *= srec.attenuation * spdf / pdf_val;
            
        cur = scattered;
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

__global__ void render_kernel(unsigned char* image, int width, int height, 
    camera cam, scene_object* device_objects, int num_objects, 
    texture_data* device_textures, int num_textures,
    material_data* device_materials, int num_materials, 
    bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
    int* device_prim_indices, int num_prim_indices, 
    int* device_light_indices, int num_lights,
    curandState* rand_states = nullptr){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= width || j >= height) return;
    auto pixel_center = cam.pixel_sample_start + i * cam.viewport_pixel_u_delta + j * cam.viewport_pixel_v_delta;
    auto ray_direction = pixel_center - cam.origin;
    ray r(cam.origin, ray_direction);
    color pixel_color(0, 0, 0);
    int pixel_index = (j * width + i) * 3;
    curandState local_rand_state = rand_states[pixel_index/3];
    for(int s_i = 0; s_i < cam.sqrt_spp; ++s_i){
        for(int s_j = 0; s_j < cam.sqrt_spp; ++s_j){
            r = cam.get_ray(i, j, s_i, s_j, &local_rand_state);
            pixel_color += ray_color(r, cam.max_depth, cam.background,
                device_objects, num_objects, 
                device_textures, num_textures,
                device_materials, num_materials, 
                device_bvh_nodes, num_bvh_nodes, root_node_index, 
                device_prim_indices, num_prim_indices, device_light_indices, num_lights, &local_rand_state);
        }
    }  
    pixel_color /= (cam.sqrt_spp * cam.sqrt_spp * 1.0);   
    write_color(image, pixel_index, pixel_color);
}

void camera::render(int width, int height, 
    scene_object* host_objects, int num_objects,
    texture_data* host_textures, int num_textures,
    material_data* host_materials, int num_materials, 
    bvh_node* host_bvh_nodes, int num_bvh_nodes, int root_node_index, 
    int* prim_indices, int num_prim_indices, int* light_indices, int num_lights, int frame){

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
    texture_data* device_textures;
    cudaMalloc(&device_textures, num_textures * sizeof(texture_data));
    cudaMemcpy(device_textures, host_textures, num_textures * sizeof(texture_data), cudaMemcpyHostToDevice);
    material_data* device_materials;
    cudaMalloc(&device_materials, num_materials * sizeof(material_data));
    cudaMemcpy(device_materials, host_materials, num_materials * sizeof(material_data), cudaMemcpyHostToDevice);
    scene_object* device_objects;
    cudaMalloc(&device_objects, num_objects * sizeof(scene_object));
    cudaMemcpy(device_objects, host_objects, num_objects * sizeof(scene_object), cudaMemcpyHostToDevice);
    int* device_prim_indices;
    cudaMalloc(&device_prim_indices, num_prim_indices * sizeof(int));
    cudaMemcpy(device_prim_indices, prim_indices, num_prim_indices * sizeof(int), cudaMemcpyHostToDevice);
    bvh_node* device_bvh_nodes;
    cudaMalloc(&device_bvh_nodes, num_bvh_nodes * sizeof(bvh_node));
    cudaMemcpy(device_bvh_nodes, host_bvh_nodes, num_bvh_nodes * sizeof(bvh_node), cudaMemcpyHostToDevice);
    int* device_light_indices;
    if(num_lights > 0){
        assert(light_indices != nullptr && "Light indices must not be empty.");
        cudaMalloc(&device_light_indices, num_lights * sizeof(int));
        cudaMemcpy(device_light_indices, light_indices, num_lights * sizeof(int), cudaMemcpyHostToDevice);
    }

    render_kernel<<<grid_size, block_size>>>(image, width, height,
        *this, device_objects, num_objects, 
        device_textures, num_textures,
        device_materials, num_materials, 
        device_bvh_nodes, num_bvh_nodes, root_node_index,
        device_prim_indices, num_prim_indices, device_light_indices, num_lights,
        rand_states);
    CUDA_CHECK(cudaGetLastError());

    // Copy the image back to the host
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> host_image(image_size);
    cudaMemcpy(host_image.data(), image, image_size, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(device_objects);
    cudaFree(device_textures);
    cudaFree(device_materials);
    cudaFree(device_bvh_nodes);
    cudaFree(device_prim_indices);
    cudaFree(device_light_indices);
    cudaFree(image);
    cudaFree(rand_states);

    std::ostringstream name;
    name << "build/frames/frame_" << std::setw(4) << std::setfill('0') << frame << ".ppm";
    std::ofstream out(name.str());
    write_image(out, host_image, width, height);
}

void render_scene(double t, int frame){
    // Material
    const int num_materials = 5;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(.65, .05, .05)}; // red
    host_materials[1] = material_data{material_type::LAMBERTIAN, color(0.73, 0.73, 0.73)}; // white
    host_materials[2] = material_data{material_type::LAMBERTIAN, color(0.12, .45, .15)}; // green
    host_materials[3] = material_data{material_type::DIFFUSE_LIGHT, color(15, 15, 15)}; // light
    host_materials[4] = material_data{material_type::DIELECTRIC, color(1.0, 1.0, 1.0), /*texture_id=*/-1, /*fuzz=*/0.0, /*refraction_index=*/1.5}; // glass

    // Objects
    const int num_objects = 8;
    scene_object host_objects[num_objects];
    host_objects[0].type = object_type::QUAD;
    host_objects[0].quad_data = quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), /*material_id=*/2);
    host_objects[1].type = object_type::QUAD;
    host_objects[1].quad_data = quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), /*material_id=*/0);
    host_objects[2].type = object_type::QUAD;
    host_objects[2].quad_data = quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), /*material_id=*/3);
    host_objects[3].type = object_type::QUAD;
    host_objects[3].quad_data = quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), /*material_id=*/1);
    host_objects[4].type = object_type::QUAD;
    host_objects[4].quad_data = quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), /*material_id=*/1);
    host_objects[5].type = object_type::QUAD;
    host_objects[5].quad_data = quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), /*material_id=*/1);
    host_objects[6].type = object_type::BOX;
    host_objects[6].box_data = box(point3(0, 0, 0), point3(165, 330, 165), /*material_id=*/1, /*angle=*/15, /*offset=*/vec3(265, 0, 295));
    // host_objects[7].type = object_type::BOX;
    // host_objects[7].box_data = box(point3(0, 0, 0), point3(165, 165, 165), /*material_id=*/1, /*angle=*/-18, /*offset=*/vec3(130, 0, 65));
    host_objects[7].type = object_type::SPHERE;
    host_objects[7].sphere_data = sphere(point3(190, 90, 190), 90, /*material_id=*/4);

    // lights
    const int num_lights = 2;
    int light_indices[num_lights] = {2, 7};
   
    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // animate camera
    double angle = t * 0.5;
    double radius = 800.0;
    point3 eye(278 + radius * sin(angle), 278, -radius * cos(angle));

    camera cam;
    cam.init(/*image_width=*/600, /*samples_per_pixel=*/100, /*max_depth=*/50, 
        /*aspect_ratio=*/1.0, /*vfov=*/40, 
        /*lookfrom=*/eye, /*lookat=*/point3(278, 278, 0), /*vup=*/vec3(0, 1, 0));
    cam.render(cam.image_width, cam.image_height, 
        host_objects, num_objects,
        /*host_textures=*/nullptr, /*num_textures=*/0,
        host_materials, num_materials,
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects, light_indices, num_lights, /*frame=*/frame);
}

int main(){
    int num_frames = 60;
    for(int frame = 0; frame < num_frames; ++frame) {
        double t = frame / 30.0;
        render_scene(t, frame);
        std::clog << "\rframe " << frame << " done" << std::flush;
    }
    return 0;
}