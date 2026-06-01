#include "color.h"
#include "camera.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include "material.h"
#include "bvh.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

D color ray_color(const ray& r, int depth, 
    const color& background,
    const scene_object* device_objects, int num_objects, 
    const texture_data* device_textures, int num_textures,
    const material_data* device_materials, int num_materials, 
    bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
    int* device_prim_indices, int num_prim_indices,
    curandState* state){
    hit_record rec;
    color attenuation(1.0, 1.0, 1.0);
    ray cur = r;
    while(0 < depth--){
        bool hit_anything = false;
        double closest_so_far = infinity;
        if(hit_bvh(cur, interval(0.001, closest_so_far), rec, device_bvh_nodes, device_prim_indices, device_objects, state)){
            hit_anything = true;
            closest_so_far = rec.t;
        }
        if(hit_anything){
            ray scattered;
            color scattered_attenuation;
            color color_from_emission = emitted(rec.u, rec.v, rec.p, device_textures, device_materials[rec.material_id]);
            if(scatter(cur, rec, scattered_attenuation, scattered, 
                device_textures,
                device_materials[rec.material_id], state)){
                attenuation = attenuation * scattered_attenuation + color_from_emission;
                cur = scattered;
            }else{
                return attenuation * color_from_emission;
            }
        }else{
            return attenuation * background;
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

__global__ void render_kernel(unsigned char* image, int width, int height, 
    camera cam, scene_object* device_objects, int num_objects, 
    texture_data* device_textures, int num_textures,
    material_data* device_materials, int num_materials, 
    bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
    int* device_prim_indices, int num_prim_indices,
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
                device_prim_indices, num_prim_indices, &local_rand_state);
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
    int* prim_indices, int num_prim_indices){

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

    render_kernel<<<grid_size, block_size>>>(image, width, height,
        *this, device_objects, num_objects, 
        device_textures, num_textures,
        device_materials, num_materials, 
        device_bvh_nodes, num_bvh_nodes, root_node_index,
        device_prim_indices, num_prim_indices, 
        rand_states);
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

void bounding_spheres(){
    // Texture
    const int num_textures = 1;
    texture_data host_textures[num_textures];
    host_textures[0] = texture_data{texture_type::CHECKER, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9), /*inv_scale=*/1/0.32};
    // Material
    const int num_materials = 488;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(0.5, 0.5, 0.5), /*texture_id=*/0}; // ground
    const int num_objects = 488;
    scene_object host_objects[num_objects];
    host_objects[0].type = object_type::SPHERE;
    host_objects[0].sphere_data = sphere(point3(0, -1000, 0), 1000.0, /*material_id=*/0);
    int host_material_index = 1;
    int host_object_index = 1;
    for(int a = -11; a < 11; ++a){
        for(int b = -11; b < 11; ++b){
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());
            point3 center2 = center;
            if((center-point3(4, 0.2, 0)).length() > 0.9){
                if(choose_mat < 0.8){
                    auto albedo = color::random() * color::random();
                    center2 = center + vec3(0, random_double(0, 0.5), 0);
                    host_materials[host_material_index++] = material_data{material_type::LAMBERTIAN, albedo};
                }else if(choose_mat < 0.95){
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    host_materials[host_material_index++] = material_data{material_type::METAL, albedo, /*texture_id=*/-1, fuzz};
                }else{
                    host_materials[host_material_index++] = material_data{material_type::DIELECTRIC, color(1.0, 1.0, 1.0), /*texture_id=*/-1, /*fuzz=*/0.0, /*refraction_index=*/1.5};
                }
                host_objects[host_object_index++] = scene_object{object_type::SPHERE, sphere(center, center2, 0.2, host_material_index - 1)};
            }
        }
    }

    host_materials[host_material_index++] = material_data{material_type::DIELECTRIC, color(.0, 1.0, 1.0), /*texture_id=*/-1, /*fuzz=*/0.0, /*refraction_index=*/1.5};
    host_objects[host_object_index++] = scene_object{object_type::SPHERE, sphere(point3(0, 1, 0), 1.0, host_material_index - 1)};
    host_materials[host_material_index++] = material_data{material_type::LAMBERTIAN, color(0.4, 0.2, 0.1)};
    host_objects[host_object_index++] = scene_object{object_type::SPHERE, sphere(point3(-4, 1, 0), 1.0, host_material_index - 1)};
    host_materials[host_material_index++] = material_data{material_type::METAL, color(0.7, 0.6, 0.5), /*texture_id=*/-1, /*fuzz=*/0.0};
    host_objects[host_object_index++] = scene_object{object_type::SPHERE, sphere(point3(4, 1, 0), 1.0, host_material_index - 1)};

    // BVH
    int actual_num_objects = host_object_index;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // camera
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    double vfov = 20;
    camera cam;
    cam.init(image_width, /*samples_per_pixel=*/100, /*max_depth=*/50, aspect_ratio, vfov, 
        /*lookfrom=*/point3(13, 2, 3), /*lookat=*/point3(0, 0, 0), /*vup=*/vec3(0, 1, 0), /*defocus_angle=*/0.6, /*focus_dist=*/10);
    int image_height = cam.image_height;
    
    cam.render(image_width, image_height, 
        host_objects, actual_num_objects,
        host_textures, num_textures,
        host_materials, actual_num_objects,
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void checker_spheres(){
    // Texture
    const int num_textures = 1;
    texture_data host_textures[num_textures];
    host_textures[0] = texture_data{texture_type::CHECKER, color(0.2, 0.3, 0.1), color(0.9,0.9,0.9), /*inv_scale=*/1/0.32};
    // Material
    const int num_materials = 1;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(0.5, 0.5, 0.5), /*texture_id=*/0}; // ground
    // Objects
    const int num_objects = 2;
    scene_object host_objects[num_objects];
    host_objects[0] = scene_object{object_type::SPHERE, sphere(point3(0, -10, 0), 10.0, /*material_id=*/0)};
    host_objects[1] = scene_object{object_type::SPHERE, sphere(point3(0, 10, 0), 10.0, /*material_id=*/0)};
    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // camera
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    double vfov = 20;
    camera cam;
    cam.init(image_width, /*samples_per_pixel=*/100, /*max_depth=*/50, aspect_ratio, vfov, 
        /*lookfrom=*/point3(13, 2, 3), /*lookat=*/point3(0, 0, 0), /*vup=*/vec3(0, 1, 0));
    int image_height = cam.image_height;

    cam.render(image_width, image_height, 
        host_objects, num_objects,
        host_textures, num_textures,
        host_materials, num_materials,
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);

}

void earth(){
    // Texture
    const int num_textures = 1;
    texture_data host_textures[num_textures];
    rtw_image earth_img("earthmap.jpg");
    int img_bytes = earth_img.width() * earth_img.height() * 3;
    unsigned char* device_image_data;
    cudaMalloc(&device_image_data, img_bytes);
    cudaMemcpy(device_image_data, earth_img.pixel_data(0,0),
        img_bytes, cudaMemcpyHostToDevice);
    host_textures[0] = texture_data{texture_type::IMAGE, 
        color(0, 0, 0), color(0, 0, 0), /*inv_scale=*/1.0, device_image_data, earth_img.width(), earth_img.height()};

    // Material
    const int num_materials = 1;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(1, 1, 1), /*texture_id=*/0};

    // Objects
    const int num_objects = 1;
    scene_object host_objects[num_objects];
    host_objects[0] = scene_object{object_type::SPHERE, sphere(point3(0, 0, 0), 2, /*material_id=*/0)};

    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // camera
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    double vfov = 20;
    camera cam;
    cam.init(image_width, /*samples_per_pixel=*/100, /*max_depth=*/50, aspect_ratio, vfov, 
        /*lookfrom=*/point3(0, 0, 12), /*lookat=*/point3(0, 0, 0), /*vup=*/vec3(0, 1, 0));
    int image_height = cam.image_height;

    cam.render(image_width, image_height, 
        host_objects, num_objects,
        host_textures, num_textures,
        host_materials, num_materials, 
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void perlin_spheres(){
    // Texture
    const int num_textures = 1;
    texture_data host_textures[num_textures];
    // host_textures[0] = texture_data{
    //     .type = texture_type::NOISE,
    //     .inv_scale = 1.0,
    //     .noise = perlin(),
    // };
    host_textures[0].type = texture_type::NOISE;
    host_textures[0].inv_scale = 1.0;
    host_textures[0].noise_scale = 4.0;
    // Material
    const int num_materials = 1;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(1, 1, 1), /*texture_id=*/0};
    // Objects
    const int num_objects = 2;
    scene_object host_objects[num_objects];
    host_objects[0] = scene_object{object_type::SPHERE, sphere(point3(0, -1000, 0), 1000.0, /*material_id=*/0)};
    host_objects[1] = scene_object{object_type::SPHERE, sphere(point3(0, 2, 0), 2, /*material_id=*/0)};
    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // camera
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 800;
    double vfov = 20;
    camera cam;
    cam.init(image_width, /*samples_per_pixel=*/100, /*max_depth=*/50, aspect_ratio, vfov, 
        /*lookfrom=*/point3(13, 2, 3), /*lookat=*/point3(0, 0, 0), /*vup=*/vec3(0, 1, 0));
    int image_height = cam.image_height;

    cam.render(image_width, image_height, 
        host_objects, num_objects,
        host_textures, num_textures, 
        host_materials, num_materials, 
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void quads() {
    // Material
    const int num_materials = 5;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(1.0, 0.2, 0.2)};
    host_materials[1] = material_data{material_type::LAMBERTIAN, color(0.2, 1.0, 0.2)};
    host_materials[2] = material_data{material_type::LAMBERTIAN, color(0.2, 0.2, 1.0)};
    host_materials[3] = material_data{material_type::LAMBERTIAN, color(1.0, 0.5, 0.0)};
    host_materials[4] = material_data{material_type::LAMBERTIAN, color(0.2, 0.8, 0.8)};
    host_materials[5] = material_data{material_type::ISOTROPIC, color(1.0, 1.0, 1.0)}; // white smoke
    host_materials[6] = material_data{material_type::ISOTROPIC, color(0.0, 0.0, 0.0)}; // black smoke
    // Objects
    const int num_objects = 5;
    scene_object host_objects[num_objects];
    host_objects[0].type = object_type::QUAD;
    host_objects[0].quad_data = quad(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), /*material_id=*/0);
    host_objects[1].type = object_type::QUAD;
    host_objects[1].quad_data = quad(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), /*material_id=*/1);
    host_objects[2].type = object_type::QUAD;
    host_objects[2].quad_data = quad(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), /*material_id=*/2);
    host_objects[3].type = object_type::QUAD;
    host_objects[3].quad_data = quad(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), /*material_id=*/3);
    host_objects[4].type = object_type::QUAD;
    host_objects[4].quad_data = quad(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), /*material_id=*/4);

    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // camera
    camera cam;
    cam.init(/*image_width=*/800, /*samples_per_pixel=*/100, /*max_depth=*/50, 
        /*aspect_ratio=*/1.0, /*vfov=*/80, 
        /*lookfrom=*/point3(0, 0, 9), /*lookat=*/point3(0, 0, 0), /*vup=*/vec3(0, 1, 0));
    cam.render(cam.image_width, cam.image_height, 
        host_objects, num_objects,
        /*host_textures=*/nullptr, /*num_textures=*/0,
        host_materials, num_materials,
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void simple_light(){
    // Texture
    const int num_textures = 2;
    texture_data host_textures[num_textures];
    host_textures[0].type = texture_type::NOISE;
    host_textures[0].noise_scale = 4.0;
    host_textures[1].type = texture_type::SOLID_COLOR;
    host_textures[1].color1 = color(4, 4, 4);
    
    // Material
    const int num_materials = 2;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(1, 1, 1), /*texture_id=*/0};
    host_materials[1] = material_data{material_type::DIFFUSE_LIGHT, color(0, 0, 0), /*texture_id=*/1};

    // Objects
    const int num_objects = 4;
    scene_object host_objects[num_objects];
    host_objects[0].type = object_type::SPHERE;
    host_objects[0].sphere_data = sphere(point3(0, -1000, 0), 1000.0, /*material_id=*/0);
    host_objects[1].type = object_type::SPHERE;
    host_objects[1].sphere_data = sphere(point3(0, 2, 0), 2.0, /*material_id=*/0);
    host_objects[2].type = object_type::SPHERE;
    host_objects[2].sphere_data = sphere(point3(0, 7, 0), 2.0, /*material_id=*/1);
    host_objects[3].type = object_type::QUAD;
    host_objects[3].quad_data = quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), /*material_id=*/1);

    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    // camera
    camera cam;
    cam.init(/*image_width=*/800, /*samples_per_pixel=*/100, /*max_depth=*/50, 
        /*aspect_ratio=*/16.0/9.0, /*vfov=*/20, 
        /*lookfrom=*/point3(26, 3, 6), /*lookat=*/point3(0, 2, 0), /*vup=*/vec3(0, 1, 0));
    cam.render(cam.image_width, cam.image_height, 
        host_objects, num_objects,
        host_textures, num_textures,
        host_materials, num_materials, 
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void cornell_box(){
    // Material
    const int num_materials = 4;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(.65, .05, .05)}; // red
    host_materials[1] = material_data{material_type::LAMBERTIAN, color(0.73, 0.73, 0.73)}; // white
    host_materials[2] = material_data{material_type::LAMBERTIAN, color(0.12, .45, .15)}; // green
    host_materials[3] = material_data{material_type::DIFFUSE_LIGHT, color(15, 15, 15)}; // light

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
    host_objects[7].type = object_type::BOX;
    host_objects[7].box_data = box(point3(0, 0, 0), point3(165, 165, 165), /*material_id=*/1, /*angle=*/-18, /*offset=*/vec3(130, 0, 65));
    // int object_index = 6;
    // add_box_transformed(host_objects, object_index, point3(0, 0, 0), point3(165, 330, 165), /*material_id=*/1, 15, vec3(265, 0, 295));
    // add_box_transformed(host_objects, object_index, point3(0, 0, 0), point3(165, 165, 165), /*material_id=*/1, -18, vec3(130, 0, 65));
   
    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    camera cam;
    cam.init(/*image_width=*/600, /*samples_per_pixel=*/196, /*max_depth=*/50, 
        /*aspect_ratio=*/1.0, /*vfov=*/40, 
        /*lookfrom=*/point3(278, 278, -800), /*lookat=*/point3(278, 278, 0), /*vup=*/vec3(0, 1, 0));
    cam.render(cam.image_width, cam.image_height, 
        host_objects, num_objects,
        /*host_textures=*/nullptr, /*num_textures=*/0,
        host_materials, num_materials,
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void cornell_smoke(){
    // Material
    const int num_materials = 6;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(.65, .05, .05)}; // red
    host_materials[1] = material_data{material_type::LAMBERTIAN, color(0.73, 0.73, 0.73)}; // white
    host_materials[2] = material_data{material_type::LAMBERTIAN, color(0.12, .45, .15)}; // green
    host_materials[3] = material_data{material_type::DIFFUSE_LIGHT, color(7, 7, 7)}; // light
    host_materials[4] = material_data{material_type::ISOTROPIC, color(1.0, 1.0, 1.0)}; // white smoke
    host_materials[5] = material_data{material_type::ISOTROPIC, color(0.0, 0.0, 0.0)}; // black smoke

    // Objects
    const int num_objects = 8;
    scene_object host_objects[num_objects];
    host_objects[0].type = object_type::QUAD;
    host_objects[0].quad_data = quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), /*material_id=*/2);
    host_objects[1].type = object_type::QUAD;
    host_objects[1].quad_data = quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), /*material_id=*/0);
    host_objects[2].type = object_type::QUAD;
    host_objects[2].quad_data = quad(point3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), /*material_id=*/3);
    host_objects[3].type = object_type::QUAD;
    host_objects[3].quad_data = quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), /*material_id=*/1);
    host_objects[4].type = object_type::QUAD;
    host_objects[4].quad_data = quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), /*material_id=*/1);
    host_objects[5].type = object_type::QUAD;
    host_objects[5].quad_data = quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), /*material_id=*/1);
    auto box1 = box(point3(0, 0, 0), point3(165, 330, 165), /*material_id=*/1, /*angle=*/15, /*offset=*/vec3(265, 0, 295));
    auto box2 = box(point3(0, 0, 0), point3(165, 165, 165), /*material_id=*/1, /*angle=*/-18, /*offset=*/vec3(130, 0, 65));
    host_objects[6].type = object_type::CONSTANT_MEDIUM;
    host_objects[6].constant_medium_data = constant_medium(box1, 0.01, /*material_id=*/5);
    host_objects[7].type = object_type::CONSTANT_MEDIUM;
    host_objects[7].constant_medium_data = constant_medium(box2, 0.01, /*material_id=*/4);
    // BVH
    int actual_num_objects = num_objects;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);

    camera cam;
    cam.init(/*image_width=*/600, /*samples_per_pixel=*/200, /*max_depth=*/50, 
        /*aspect_ratio=*/1.0, /*vfov=*/40, 
        /*lookfrom=*/point3(278, 278, -800), /*lookat=*/point3(278, 278, 0), /*vup=*/vec3(0, 1, 0));
    cam.render(cam.image_width, cam.image_height, 
        host_objects, num_objects,
        /*host_textures=*/nullptr, /*num_textures=*/0,
        host_materials, num_materials,
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);
}

void final_scene(int image_width, int samples_per_pixel, int max_depth){
    // Texture
    const int num_textures = 2;
    texture_data host_textures[num_textures];
    host_textures[0].type = texture_type::IMAGE;
    rtw_image earth_img("earthmap.jpg");
    int img_bytes = earth_img.width() * earth_img.height() * 3;
    unsigned char* device_image_data;
    cudaMalloc(&device_image_data, img_bytes);
    cudaMemcpy(device_image_data, earth_img.pixel_data(0,0),
        img_bytes, cudaMemcpyHostToDevice);
    host_textures[0].type = texture_type::IMAGE;
    host_textures[0].image_data = device_image_data;
    host_textures[0].image_width = earth_img.width();
    host_textures[0].image_height = earth_img.height();
    host_textures[1].type = texture_type::NOISE;
    host_textures[1].noise_scale = 0.2;
    
    // Material
    const int num_materials = 10;
    material_data host_materials[num_materials];
    host_materials[0] = material_data{material_type::LAMBERTIAN, color(0.48, 0.83, 0.53)}; // ground
    host_materials[1] = material_data{material_type::DIFFUSE_LIGHT, color(15, 15, 15)}; // light
    host_materials[2] = material_data{material_type::LAMBERTIAN, color(0.7, 0.3, 0.1)}; // sphere
    host_materials[3] = material_data{material_type::DIELECTRIC, color(1.0, 1.0, 1.0), /*texture_id=*/-1, /*fuzz=*/0.0, /*refraction_index=*/1.5}; // dielectric
    host_materials[4] = material_data{material_type::METAL, color(0.8, 0.8, 0.9), /*texture_id=*/-1, /*fuzz=*/1.0}; // metal
    host_materials[5] = material_data{material_type::ISOTROPIC, color(1.0, 1.0, 1.0)}; // white smoke
    host_materials[6] = material_data{material_type::ISOTROPIC, color(0.2, 0.4, 0.9)}; // blue smoke
    host_materials[7] = material_data{material_type::LAMBERTIAN, color(1.0, 1.0, 1.0), /*texture_id=*/0}; // earth
    host_materials[8] = material_data{material_type::LAMBERTIAN, color(1.0, 1.0, 1.0), /*texture_id=*/1}; // noise texture
    host_materials[9] = material_data{material_type::LAMBERTIAN, color(0.73, 0.73, 0.73)}; // white

    // Objects
    const int num_objects = 400 + 9 + 1000;
    scene_object host_objects[num_objects];
    int object_index = 0;
    int boxes_per_side = 20;
    for(int i = 0; i < boxes_per_side; ++i){
        for(int j = 0; j < boxes_per_side; ++j){
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0+w;
            auto z1 = z0+w;
            auto y1 = random_double(1, 101);
            host_objects[object_index].type = object_type::BOX;
            host_objects[object_index++].box_data = box(point3(x0, y0, z0), point3(x1, y1, z1), /*material_id=*/0);
        }
    }
    host_objects[object_index].type = object_type::QUAD;
    host_objects[object_index++].quad_data = quad(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), /*material_id=*/1);
    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    host_objects[object_index].type = object_type::SPHERE;
    host_objects[object_index++].sphere_data = sphere(center1, center2, 50, /*material_id=*/2);
    host_objects[object_index].type = object_type::SPHERE;
    host_objects[object_index++].sphere_data = sphere(point3(260, 150, 45), 50, /*material_id=*/3);
    host_objects[object_index].type = object_type::SPHERE;
    host_objects[object_index++].sphere_data = sphere(point3(0, 150, 145), 50, /*material_id=*/4);

    // boundary
    auto boundary = sphere(point3(360, 150, 145), 70, /*material_id=*/3);
    host_objects[object_index].type = object_type::SPHERE;
    host_objects[object_index++].sphere_data = boundary;
    host_objects[object_index].type = object_type::CONSTANT_MEDIUM;
    host_objects[object_index++].constant_medium_data = constant_medium(boundary, 0.0001, /*material_id=*/5);

    // earth
    host_objects[object_index].type = object_type::SPHERE;
    host_objects[object_index++].sphere_data = sphere(point3(400, 200, 400), 100, /*material_id=*/7);
    
    // perlin sphere
    host_objects[object_index].type = object_type::SPHERE;
    host_objects[object_index++].sphere_data = sphere(point3(220, 280, 300), 80, /*material_id=*/8);
    

    int ns = 1000;
    for(int i = 0; i < ns; ++i){
        point3 c_local = point3::random(0, 165);
        point3 c_world = rotate_y(c_local, 15) + vec3(-100, 270, 395);
        host_objects[object_index].type = object_type::SPHERE;
        host_objects[object_index++].sphere_data = sphere(c_world, 10, /*material_id=*/9);
    }
    
    // BVH
    int actual_num_objects = object_index;
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2*actual_num_objects-1);
    int root_node_index = build_bvh(host_bvh_nodes, prim_indices, host_objects, 0, actual_num_objects);
    
    // camera
    camera cam;
    cam.init(/*image_width=*/image_width, /*samples_per_pixel=*/samples_per_pixel, /*max_depth=*/max_depth, 
        /*aspect_ratio=*/1.0, /*vfov=*/40, 
        /*lookfrom=*/point3(478, 278, -600), /*lookat=*/point3(278, 278, 0), /*vup=*/vec3(0, 1, 0));
    cam.render(cam.image_width, cam.image_height, 
        host_objects, actual_num_objects,
        host_textures, num_textures,
        host_materials, num_materials, 
        host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index, 
        prim_indices.data(), actual_num_objects);


}

int main(){
    switch(7){
        case 1: bounding_spheres(); break;
        case 2: checker_spheres(); break;
        case 3: earth(); break;
        case 4: perlin_spheres(); break;
        case 5: quads(); break;
        case 6: simple_light(); break;
        case 7: cornell_box(); break;
        case 8: cornell_smoke(); break;
        case 9: final_scene(800, 10000, 40); break;
    }
    return 0;
}