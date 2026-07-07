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
#include <unordered_map>
#include <string>

#include <fstream>
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
                mixture_pdf_generate(rec.p, rec.normal, device_objects, device_light_indices,
                                     num_lights, srec.pdf_type, state),
                cur.time());
        pdf_val = mixture_pdf_value(rec.p, scattered.direction(), rec.normal, device_objects,
                                    device_light_indices, num_lights, srec.pdf_type);

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

__global__ void render_kernel(unsigned char* image, int width, int height, camera cam,
                              scene_object* device_objects, int num_objects,
                              texture_data* device_textures, int num_textures,
                              material_data* device_materials, int num_materials,
                              bvh_node* device_bvh_nodes, int num_bvh_nodes, int root_node_index,
                              int* device_prim_indices, int num_prim_indices,
                              int* device_light_indices, int num_lights,
                              curandState* rand_states = nullptr) {
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
    for (int s_i = 0; s_i < cam.sqrt_spp; ++s_i) {
        for (int s_j = 0; s_j < cam.sqrt_spp; ++s_j) {
            r = cam.get_ray(i, j, s_i, s_j, &local_rand_state);
            pixel_color +=
                ray_color(r, cam.max_depth, cam.background, device_objects, num_objects,
                          device_textures, num_textures, device_materials, num_materials,
                          device_bvh_nodes, num_bvh_nodes, root_node_index, device_prim_indices,
                          num_prim_indices, device_light_indices, num_lights, &local_rand_state);
        }
    }
    pixel_color /= (cam.sqrt_spp * cam.sqrt_spp * 1.0);
    write_color(image, pixel_index, pixel_color);
}

void camera::render(int width, int height, scene_object* host_objects, int num_objects,
                    texture_data* host_textures, int num_textures, material_data* host_materials,
                    int num_materials, bvh_node* host_bvh_nodes, int num_bvh_nodes,
                    int root_node_index, int* prim_indices, int num_prim_indices,
                    int* light_indices, int num_lights, int frame) {
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
    int* device_light_indices;
    if (num_lights > 0) {
        assert(light_indices != nullptr && "Light indices must not be empty.");
        cudaMalloc(&device_light_indices, num_lights * sizeof(int));
        cudaMemcpy(device_light_indices, light_indices, num_lights * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    render_kernel<<<grid_size, block_size>>>(
        image, width, height, *this, device_objects, num_objects, device_textures, num_textures,
        device_materials, num_materials, device_bvh_nodes, num_bvh_nodes, root_node_index,
        device_prim_indices, num_prim_indices, device_light_indices, num_lights, rand_states);
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

void primitive_test_scene(double t, int frame) {
    // Materials
    std::vector<material_data> host_materials;

    int ground_mat = host_materials.size();
    host_materials.push_back(material_data{material_type::LAMBERTIAN, color(0.55, 0.55, 0.55)});

    int cylinder_mat = host_materials.size();
    host_materials.push_back(material_data{material_type::LAMBERTIAN, color(0.85, 0.20, 0.15)});

    int cone_mat = host_materials.size();
    host_materials.push_back(material_data{material_type::LAMBERTIAN, color(0.15, 0.35, 0.90)});

    int light_mat = host_materials.size();
    host_materials.push_back(material_data{material_type::DIFFUSE_LIGHT, color(8.0, 8.0, 8.0)});

    // Objects
    std::vector<scene_object> host_objects;
    std::vector<int> light_indices;

    // Ground
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(-5, 0, -5), vec3(10, 0, 0), vec3(0, 0, 10), ground_mat);

    // Area light above the objects
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(-2, 5, -2), vec3(4, 0, 0), vec3(0, 0, 4), light_mat);
    light_indices.push_back(host_objects.size() - 1);

    // Vertical cylinder on the left
    host_objects.push_back({});
    host_objects.back().type = object_type::CYLINDER;
    host_objects.back().cylinder_data = cylinder(/*top_center=*/point3(-1.5, 2.5, 0),
                                                 /*base_center=*/point3(-1.5, 0.0, 0),
                                                 /*radius=*/0.45,
                                                 /*material_id=*/cylinder_mat);

    // Vertical cone on the right
    host_objects.push_back({});
    host_objects.back().type = object_type::CONE;
    host_objects.back().cone_data = cone(/*apex=*/point3(1.5, 2.5, 0),
                                         /*base_center=*/point3(1.5, 0.0, 0),
                                         /*radius=*/0.65,
                                         /*material_id=*/cone_mat);

    // Optional: tilted cylinder to test arbitrary-axis cylinder
    host_objects.push_back({});
    host_objects.back().type = object_type::CYLINDER;
    host_objects.back().cylinder_data = cylinder(/*top_center=*/point3(0.0, 2.8, -1.4),
                                                 /*base_center=*/point3(-0.8, 0.4, -1.4),
                                                 /*radius=*/0.25,
                                                 /*material_id=*/cylinder_mat);

    // BVH
    int actual_num_objects = host_objects.size();

    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);

    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2 * actual_num_objects - 1);

    int root_node_index =
        build_bvh(host_bvh_nodes, prim_indices, host_objects.data(), 0, actual_num_objects);

    // Camera
    camera cam;
    cam.init(/*image_width=*/800,
             /*samples_per_pixel=*/100,
             /*max_depth=*/20,
             /*aspect_ratio=*/16.0 / 9.0,
             /*vfov=*/35,
             /*lookfrom=*/point3(0, 2.0, 7.0),
             /*lookat=*/point3(0, 1.2, 0),
             /*vup=*/vec3(0, 1, 0),
             /*defocus_angle=*/0.0,
             /*focus_dist=*/10.0,
             /*background=*/color(0.02, 0.03, 0.05));

    cam.render(cam.image_width, cam.image_height, host_objects.data(), actual_num_objects,
               /*host_textures=*/nullptr, /*num_textures=*/0, host_materials.data(),
               host_materials.size(), host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index,
               prim_indices.data(), actual_num_objects, light_indices.data(), light_indices.size(),
               /*frame=*/frame);
}


void render_window_tree_scene(double t, int frame) {
    std::vector<material_data> host_materials;
    std::unordered_map<std::string, int> material_id_map;

    auto add_material = [&](const std::string& name, const material_data& mat) {
        material_id_map[name] = static_cast<int>(host_materials.size());
        host_materials.push_back(mat);
    };

    // ------------------------------------------------------------------
    // Materials
    // Adjust constructor fields if your material_data definition differs.
    // ------------------------------------------------------------------

    add_material("grass",
        material_data{material_type::LAMBERTIAN, color(0.10, 0.20, 0.10)});

    add_material("fence",
        material_data{material_type::LAMBERTIAN, color(0.38, 0.30, 0.18)});

    add_material("bark",
        material_data{material_type::LAMBERTIAN, color(0.30, 0.22, 0.12)});

    add_material("leaf",
        material_data{material_type::LAMBERTIAN, color(0.08, 0.22, 0.10)});

    add_material("lamp_post",
        material_data{material_type::LAMBERTIAN, color(0.16, 0.16, 0.18)});

    add_material("mountain",
        material_data{material_type::LAMBERTIAN, color(0.05, 0.06, 0.08)});

    add_material("window_frame",
        material_data{material_type::LAMBERTIAN, color(0.22, 0.18, 0.12)});

    add_material("lamp_light",
        material_data{material_type::DIFFUSE_LIGHT, color(8.0, 7.2, 5.8)});

    add_material("moon_light",
        material_data{material_type::DIFFUSE_LIGHT, color(1.8, 1.9, 2.5)});

    // Replace this with your exact dielectric constructor format.
    add_material("window_glass",
        material_data{
            material_type::DIELECTRIC,
            color(1.0, 1.0, 1.0),
            -1,
            0.0,
            1.5
        });

    // ------------------------------------------------------------------
    // Scene objects
    // ------------------------------------------------------------------

    std::vector<scene_object> host_objects;
    std::vector<int> light_indices;

    auto push_quad = [&](const point3& q, const vec3& u, const vec3& v, const std::string& mat) {
        host_objects.push_back({});
        host_objects.back().type = object_type::QUAD;
        host_objects.back().quad_data = quad(q, u, v, material_id_map.at(mat));
    };

    auto push_box = [&](const point3& a, const point3& b, const std::string& mat) {
        host_objects.push_back({});
        host_objects.back().type = object_type::BOX;
        host_objects.back().box_data = box(a, b, material_id_map.at(mat));
    };

    auto push_sphere = [&](const point3& c, double r, const std::string& mat, bool is_light=false) {
        host_objects.push_back({});
        host_objects.back().type = object_type::SPHERE;
        host_objects.back().sphere_data = sphere(c, r, material_id_map.at(mat));
        if (is_light) {
            light_indices.push_back(static_cast<int>(host_objects.size()) - 1);
        }
    };

    auto push_cylinder = [&](const point3& top_center, const point3& base_center,
                             double r, const std::string& mat) {
        host_objects.push_back({});
        host_objects.back().type = object_type::CYLINDER;
        host_objects.back().cylinder_data =
            cylinder(top_center, base_center, r, material_id_map.at(mat));
    };

    auto push_cone = [&](const point3& apex, const point3& base_center,
                         double r, const std::string& mat) {
        host_objects.push_back({});
        host_objects.back().type = object_type::CONE;
        host_objects.back().cone_data =
            cone(apex, base_center, r, material_id_map.at(mat));
    };

    // ------------------------------------------------------------------
    // Coordinate plan
    // camera roughly around z = 0, looking +z
    // window just in front of camera
    // outdoor scene farther in +z
    // ------------------------------------------------------------------

    const double ground_y = 0.0;
    const double window_z = 2.0;

    // ------------------------------------------------
    // 1. Ground / garden
    // ------------------------------------------------
    push_quad(
        point3(-14.0, ground_y, 4.0),
        vec3(28.0, 0.0, 0.0),
        vec3(0.0, 0.0, 42.0),
        "grass"
    );

    // ------------------------------------------------
    // 2. Window glass: use one QUAD, not a box
    //    Important: choose u/v order so normal faces camera if needed.
    // ------------------------------------------------
    push_quad(
        point3(-5.0, 1.0, window_z),
        vec3(0.0, 7.0, 0.0),
        vec3(10.0, 0.0, 0.0),
        "window_glass"
    );

    // ------------------------------------------------
    // 3. Window frame
    // ------------------------------------------------
    // left frame
    push_box(point3(-5.25, 0.8, window_z - 0.05), point3(-5.0, 8.2, window_z + 0.05), "window_frame");
    // right frame
    push_box(point3( 5.0, 0.8, window_z - 0.05), point3( 5.25, 8.2, window_z + 0.05), "window_frame");
    // bottom frame
    push_box(point3(-5.25, 0.8, window_z - 0.05), point3( 5.25, 1.0, window_z + 0.05), "window_frame");
    // top frame
    push_box(point3(-5.25, 8.0, window_z - 0.05), point3( 5.25, 8.2, window_z + 0.05), "window_frame");

    // Optional middle divider
    push_box(point3(-0.08, 1.0, window_z - 0.05), point3(0.08, 8.0, window_z + 0.05), "window_frame");

    // ------------------------------------------------
    // 4. Fence
    // ------------------------------------------------
    double fence_z = 12.0;
    double fence_h = 1.4;

    // posts
    for (int i = 0; i < 9; ++i) {
        double x = -8.0 + i * 2.0;
        push_box(
            point3(x - 0.08, ground_y, fence_z - 0.08),
            point3(x + 0.08, ground_y + fence_h, fence_z + 0.08),
            "fence"
        );
    }

    // rails
    push_box(
        point3(-8.2, ground_y + 0.45, fence_z - 0.05),
        point3( 8.2, ground_y + 0.57, fence_z + 0.05),
        "fence"
    );

    push_box(
        point3(-8.2, ground_y + 1.00, fence_z - 0.05),
        point3( 8.2, ground_y + 1.12, fence_z + 0.05),
        "fence"
    );

    // ------------------------------------------------
    // 5. Trees
    // Use cylinder trunk + cone foliage
    // ------------------------------------------------
    auto add_tree = [&](double x, double z, double trunk_h, double trunk_r,
                        double cone_base_y, double cone_h, double cone_r) {
        push_cylinder(
            point3(x, ground_y + trunk_h, z),
            point3(x, ground_y, z),
            trunk_r,
            "bark"
        );

        // lower foliage cone
        push_cone(
            point3(x, ground_y + cone_base_y + cone_h, z),
            point3(x, ground_y + cone_base_y, z),
            cone_r,
            "leaf"
        );

        // upper foliage cone
        push_cone(
            point3(x, ground_y + cone_base_y + cone_h + 1.2, z),
            point3(x, ground_y + cone_base_y + 1.0, z),
            cone_r * 0.75,
            "leaf"
        );
    };

    add_tree(-6.0, 18.0, 2.4, 0.18, 1.6, 2.6, 1.8);
    add_tree(-1.5, 20.5, 2.8, 0.20, 1.9, 3.0, 2.0);
    add_tree( 3.5, 17.0, 2.1, 0.16, 1.4, 2.4, 1.7);
    add_tree( 7.0, 23.0, 3.0, 0.22, 2.0, 3.2, 2.2);

    // ------------------------------------------------
    // 6. Lamp on the right
    // Use cylinder post + emissive sphere
    // ------------------------------------------------
    push_cylinder(
        point3(8.5, ground_y + 4.0, 14.5),
        point3(8.5, ground_y, 14.5),
        0.10,
        "lamp_post"
    );

    push_sphere(
        point3(8.5, ground_y + 4.3, 14.5),
        0.35,
        "lamp_light",
        true
    );

    // ------------------------------------------------
    // 7. Moon in upper-left
    // Use emissive sphere so your current PDF system can sample it.
    // ------------------------------------------------
    push_sphere(
        point3(-10.0, 10.5, 36.0),
        1.2,
        "moon_light",
        true
    );

    // ------------------------------------------------
    // 8. Far mountains / silhouettes
    // Simple stylized version using large dark boxes
    // ------------------------------------------------
    push_box(
        point3(-16.0, ground_y, 30.0),
        point3( -3.0, 5.0, 46.0),
        "mountain"
    );

    push_box(
        point3( -5.0, ground_y, 32.0),
        point3(  8.0, 7.0, 47.0),
        "mountain"
    );

    push_box(
        point3(  6.0, ground_y, 31.0),
        point3( 16.0, 4.5, 46.0),
        "mountain"
    );

    // ------------------------------------------------
    // 9. Optional: a few raindrops on the window
    // Start with just a few large droplets.
    // Later you can animate them.
    // ------------------------------------------------
    push_sphere(point3(-2.8, 6.4, window_z - 0.03), 0.16, "window_glass");
    push_sphere(point3(-1.1, 4.7, window_z - 0.03), 0.12, "window_glass");
    push_sphere(point3( 1.8, 5.8, window_z - 0.03), 0.14, "window_glass");
    push_sphere(point3( 3.2, 3.6, window_z - 0.03), 0.10, "window_glass");

    // ------------------------------------------------------------------
    // BVH
    // ------------------------------------------------------------------
    int actual_num_objects = static_cast<int>(host_objects.size());

    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);

    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2 * actual_num_objects - 1);

    int root_node_index = build_bvh(
        host_bvh_nodes,
        prim_indices,
        host_objects.data(),
        0,
        actual_num_objects
    );

    // ------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------
    camera cam;
    cam.init(
        /*image_width=*/1280,
        /*samples_per_pixel=*/200,
        /*max_depth=*/30,
        /*aspect_ratio=*/16.0 / 9.0,
        /*vfov=*/40.0,
        /*lookfrom=*/point3(0.0, 4.2, -2.5),
        /*lookat=*/point3(0.0, 4.0, 16.0),
        /*vup=*/vec3(0.0, 1.0, 0.0),
        /*defocus_angle=*/0.0,
        /*focus_dist=*/18.0,
        /*background=*/color(0.04, 0.05, 0.09)
    );

    cam.render(
        cam.image_width,
        cam.image_height,
        host_objects.data(),
        actual_num_objects,
        /*host_textures=*/nullptr,
        /*num_textures=*/0,
        host_materials.data(),
        host_materials.size(),
        host_bvh_nodes.data(),
        host_bvh_nodes.size(),
        root_node_index,
        prim_indices.data(),
        actual_num_objects,
        light_indices.data(),
        light_indices.size(),
        frame
    );
}

void render_scene(double t, int frame) {
    // anchor
    point3 origin(0, 0, 0);

    // Material
    std::vector<material_data> host_materials;
    std::unordered_map<std::string, int> material_id_map;

    host_materials.push_back(
        material_data{material_type::LAMBERTIAN, color(.65, .05, .05)});  // red
    material_id_map["red"] = host_materials.size() - 1;
    host_materials.push_back(
        material_data{material_type::LAMBERTIAN, color(0.73, 0.73, 0.73)});  // white
    material_id_map["white"] = host_materials.size() - 1;
    host_materials.push_back(
        material_data{material_type::LAMBERTIAN, color(0.0, 1.0, 0.0)});  // green
    material_id_map["green"] = host_materials.size() - 1;
    host_materials.push_back(
        material_data{material_type::DIFFUSE_LIGHT, color(10, 10, 10)});  // light
    material_id_map["light"] = host_materials.size() - 1;
    host_materials.push_back(material_data{material_type::DIELECTRIC, color(1.0, 1.0, 1.0),
                                           /*texture_id=*/-1, /*fuzz=*/0.0,
                                           /*refraction_index=*/1.5});  // glass
    material_id_map["glass"] = host_materials.size() - 1;
    host_materials.push_back(material_data{material_type::DIELECTRIC, color(1.0, 1.0, 1.0),
                                           /*texture_id=*/-1, /*fuzz=*/0.0,
                                           /*refraction_index=*/1.45});  // window glass
    material_id_map["window_glass"] = host_materials.size() - 1;

    host_materials.push_back(material_data{material_type::DIFFUSE_LIGHT, color(0.08, 0.1, 0.18)});
    material_id_map["night_sky"] = host_materials.size() - 1;

    host_materials.push_back(material_data{material_type::LAMBERTIAN, color(0.18, 0.18, 0.2)});
    material_id_map["wet_road"] = host_materials.size() - 1;

    host_materials.push_back(material_data{material_type::LAMBERTIAN, color(0.1, 0.11, 0.14)});
    material_id_map["building_dark"] = host_materials.size() - 1;

    host_materials.push_back(material_data{material_type::DIFFUSE_LIGHT, color(8.0, 6.5, 3.5)});
    material_id_map["street_lamp_light"] = host_materials.size() - 1;

    host_materials.push_back(material_data{material_type::LAMBERTIAN, color(0.08, 0.08, 0.08)});
    material_id_map["lamp_post"] = host_materials.size() - 1;

    // Objects
    std::vector<scene_object> host_objects;
    std::vector<int> light_indices;
    int image_width = 1280, image_height = 720;
    int window_size_x = 856, window_size_y = window_size_x * (image_height * 1.0 / image_width);
    int window_offset_y = -image_height * 1.0 / 36;
    int from_eye_to_window_distance = 365;
    double glass_thickness = 0.05;
    double window_z = origin.z() + from_eye_to_window_distance;
    double outside_z = window_z + 1.0;

    // left wall
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(origin.x() + window_size_x / 2,
                    origin.y() - window_size_y / 2 + window_offset_y, origin.z()),
             vec3(0, window_size_y, 0), vec3(0, 0, from_eye_to_window_distance),
             /*material_id=*/material_id_map.at("green"));
    // right wall
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(origin.x() - window_size_x / 2,
                    origin.y() - window_size_y / 2 + window_offset_y, origin.z()),
             vec3(0, 0, from_eye_to_window_distance), vec3(0, window_size_y, 0),
             /*material_id=*/material_id_map.at("green"));

    // light
    double light_size_scale = 1.0;
    point3 light_size = point3(130, 0, 52) * light_size_scale;
    point3 light_center(origin.x(), origin.y() + window_size_y / 2 + window_offset_y - 1,
                        origin.z() + from_eye_to_window_distance -
                            from_eye_to_window_distance * 0.18 - light_size.z() / 2);
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data = quad(light_center + light_size / 2, vec3(-light_size.x(), 0, 0),
                                         vec3(0, 0, -light_size.z()),
                                         /*material_id=*/material_id_map.at("light"));

    light_indices.push_back(host_objects.size() - 1);

    // floor
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(origin.x() - window_size_x / 2,
                    origin.y() - window_size_y / 2 + window_offset_y, origin.z()),
             vec3(0, 0, from_eye_to_window_distance), vec3(window_size_x, 0, 0),
             /*material_id=*/material_id_map.at("white"));

    // ceiling
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(origin.x() - window_size_x / 2,
                    origin.y() + window_size_y / 2 + window_offset_y, origin.z()),
             vec3(0, 0, from_eye_to_window_distance), vec3(window_size_x, 0, 0),
             /*material_id=*/material_id_map.at("white"));
    // host_objects.push_back({});
    // host_objects.back().type = object_type::QUAD;
    // host_objects.back().quad_data = quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0),
    // /*material_id=*/material_id_map.at("white")); host_objects.push_back({});
    // host_objects.back().type = object_type::BOX;
    // host_objects.back().box_data = box(point3(0, 0, 0), point3(165, 330, 165),
    // /*material_id=*/material_id_map.at("white"), /*angle=*/15, /*offset=*/vec3(265, 0, 295));
    // host_objects.push_back({});
    // host_objects.back().type = object_type::SPHERE;
    // host_objects.back().sphere_data = sphere(point3(190, 90, 190), 90,
    // /*material_id=*/material_id_map.at("glass"));

    // window
    host_objects.push_back({});
    host_objects.back().type = object_type::BOX;
    host_objects.back().box_data =
        box(point3(origin.x() - window_size_x / 2, origin.y() - window_size_y / 2 + window_offset_y,
                   origin.z() + from_eye_to_window_distance - glass_thickness),
            point3(origin.x() + window_size_x / 2, origin.y() + window_size_y / 2 + window_offset_y,
                   origin.z() + from_eye_to_window_distance),
            /*material_id=*/material_id_map.at("window_glass"),
            /*angle=*/0,
            /*offset=*/vec3(0, 0, 0));

    // far sky
    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(origin.x() - 2000, origin.y() - 1000, window_z + 1200), vec3(0, 2000, 0),
             vec3(4000, 0, 0), material_id_map.at("night_sky"));

    // road
    double road_y = origin.y() - window_size_y / 2 + window_offset_y - 20;
    double road_z = window_z + 20;

    host_objects.push_back({});
    host_objects.back().type = object_type::QUAD;
    host_objects.back().quad_data =
        quad(point3(origin.x() - 1000, road_y, road_z), vec3(0, 0, 1400), vec3(2000, 0, 0),
             material_id_map.at("wet_road"));

    // building silhouettes
    for (int i = 0; i < 5; ++i) {
        double x0 = -700 + i * 300;
        double h = 250 + 80 * (i % 3);
        host_objects.push_back({});
        host_objects.back().type = object_type::BOX;
        host_objects.back().box_data = box(point3(x0, origin.y() - 250, window_z + 700),
                                           point3(x0 + 180, origin.y() - 250 + h, window_z + 760),
                                           material_id_map.at("building_dark"));
    }

    // street lamps
    for (int i = 0; i < 3; ++i) {
        double z = window_z + 180 + i * 250;
        double x = -250 + i * 220;
        // lamp post
        host_objects.push_back({});
        host_objects.back().type = object_type::BOX;
        host_objects.back().box_data =
            box(point3(x, road_y, z), point3(x + 12, road_y + 330, z + 12),
                material_id_map.at("lamp_post"));

        // lamp bulb
        host_objects.push_back({});
        host_objects.back().type = object_type::SPHERE;
        host_objects.back().sphere_data =
            sphere(point3(x + 6, road_y + 360, z + 6), 25, material_id_map.at("street_lamp_light"));
        light_indices.push_back(host_objects.size() - 1);
    }

    // BVH
    int actual_num_objects = host_objects.size();
    std::vector<int> prim_indices(actual_num_objects);
    std::iota(prim_indices.begin(), prim_indices.end(), 0);
    std::vector<bvh_node> host_bvh_nodes;
    host_bvh_nodes.reserve(2 * actual_num_objects - 1);
    int root_node_index =
        build_bvh(host_bvh_nodes, prim_indices, host_objects.data(), 0, actual_num_objects);

    // animate camera
    double angle = t * 0.5;
    double radius = from_eye_to_window_distance;
    point3 eye(origin.x(), origin.y() + window_offset_y, origin.z());

    camera cam;
    cam.init(/*image_width=*/image_width, /*samples_per_pixel=*/400, /*max_depth=*/50,
             /*aspect_ratio=*/16.0 / 9.0, /*vfov=*/90,
             /*lookfrom=*/eye,
             /*lookat=*/origin + vec3(0, window_offset_y, from_eye_to_window_distance),
             /*vup=*/vec3(0, 1, 0),
             /*defocus_angle=*/0.0, /*focus_dist=*/10, /*background=*/color(0.03, 0.04, 0.07));
    cam.render(cam.image_width, cam.image_height, host_objects.data(), actual_num_objects,
               /*host_textures=*/nullptr, /*num_textures=*/0, host_materials.data(),
               host_materials.size(), host_bvh_nodes.data(), host_bvh_nodes.size(), root_node_index,
               prim_indices.data(), actual_num_objects, light_indices.data(), light_indices.size(),
               /*frame=*/frame);
}

int main() {
    int num_frames = 1;
    double fps = 30.0;
    for (int frame = 0; frame < num_frames; ++frame) {
        double t = frame / fps;
        render_window_tree_scene(t, frame);
        // primitive_test_scene(t, frame);
        // render_scene(t, frame);
        std::clog << "\rframe " << frame << " done" << std::flush;
    }
    return 0;
}
