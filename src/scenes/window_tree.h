#include "../scene.h"
#include "../camera.h"
#include "../cloud.h"
#include "../lightning.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <numeric>   // for std::iota

void render_window_tree_scene(double t, int frame, const rain_settings& rain,
                              const cloud_settings& clouds = cloud_settings(),
                              const lightning_settings& lightning = lightning_settings(),
                              const render_quality_settings& quality = render_quality_settings()) {
    std::vector<material_data> host_materials;
    std::unordered_map<std::string, int> material_id_map;
    lightning_frame_state lightning_frame = lightning_at_time(t, lightning, clouds);

    auto add_material = [&](const std::string& name, const material_data& mat) {
        material_id_map[name] = static_cast<int>(host_materials.size());
        host_materials.push_back(mat);
    };

    bool use_physical_rain = rain.mode == rain_mode::PATH_TRACING || rain.mode == rain_mode::HYBRID;

    if (use_physical_rain) {
        // Add sphere/cylinder droplets here.
    }
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

    add_material("mountain_far",
        material_data{material_type::LAMBERTIAN, color(0.09, 0.11, 0.15)});

    add_material("mountain_mid",
        material_data{material_type::LAMBERTIAN, color(0.06, 0.075, 0.105)});

    add_material("mountain_near",
        material_data{material_type::LAMBERTIAN, color(0.035, 0.045, 0.065)});

    add_material("window_frame",
        material_data{material_type::LAMBERTIAN, color(0.22, 0.18, 0.12)});

    add_material("lamp_light",
        material_data{material_type::DIFFUSE_LIGHT, color(8.0, 7.2, 5.8)});

    add_material("sun_light",
        material_data{material_type::DIFFUSE_LIGHT, color(300.0, 310.0, 320.0)});

    add_material("moon_light",
        material_data{material_type::DIFFUSE_LIGHT, color(1.8, 1.9, 2.5)});
    add_material("sky_light",
        material_data{
            material_type::DIFFUSE_LIGHT,
            (clouds.enabled ? color(0.28, 0.32, 0.38) : color(2.2, 2.3, 2.4)) +
                lightning_frame.flash * lightning.sky_flash * color(0.62, 0.76, 1.0),
            -1,
            0.0,
            1.0
        });

    add_material("cloud_dark",
        material_data{material_type::HENYEY_GREENSTEIN, color(0.76, 0.79, 0.84),
                      -1, 0.0, 1.0, false, 0.72});

    add_material("cloud_light",
        material_data{material_type::HENYEY_GREENSTEIN, color(0.90, 0.92, 0.96),
                      -1, 0.0, 1.0, false, 0.72});

    add_material(
        "lightning_flash",
        material_data{material_type::DIFFUSE_LIGHT,
                      lightning_frame.flash * lightning.intensity * color(0.68, 0.82, 1.0)});

    add_material(
        "lightning_bolt",
        material_data{material_type::DIFFUSE_LIGHT,
                      lightning_frame.flash * lightning.intensity * 1.8 *
                          color(0.72, 0.86, 1.0)});

    // Replace this with your exact dielectric constructor format.
    add_material("window_glass",
        material_data{
            material_type::DIELECTRIC,
            color(1.0, 1.0, 1.0),
            -1,
            0.0,
            1.5,
            true
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

    auto push_paraboloid = [&](const point3& base_center, double radius_x,
                               double radius_z, double height, const std::string& mat) {
        host_objects.push_back({});
        host_objects.back().type = object_type::PARABOLOID;
        host_objects.back().paraboloid_data =
            paraboloid(base_center, radius_x, radius_z, height, material_id_map.at(mat));
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

    // // Optional middle divider
    // push_box(point3(-0.08, 1.0, window_z - 0.05), point3(0.08, 8.0, window_z + 0.05), "window_frame");

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
    // push_sphere(
    //     point3(-10.0, 10.5, 36.0),
    //     1.2,
    //     "moon_light",
    //     true
    // );
    // Keep this broad fill light above the distant sun. If it sits between the
    // sun and the ground, its opaque quad intercepts every sun ray and prevents
    // the moving cloud volumes from casting distinct sun shadows.
    push_quad(
        point3(-250.0, 240.0, -200.0),
        vec3(500.0, 0.0, 0.0),
        vec3(0.0, 0.0, 500.0),
        "sky_light"
    );
    light_indices.push_back(static_cast<int>(host_objects.size()) - 1);

    // ------------------------------------------------
    // 8. Moving volumetric clouds
    // A noisy heterogeneous field blends the lobes into a continuous bank
    // with cavities, wispy edges, and a comparatively flat storm-cloud base.
    // It sits below the sky light and casts real moving volumetric shadows.
    // ------------------------------------------------
    if (clouds.enabled) {
        // Keep the sun distant and nearly overhead so its rays are almost
        // parallel across the scene. Scaling its radius with the distance
        // preserves a soft solar disc without the divergent, triangular tree
        // shadows produced by a nearby point-like source.
        // The broad sky quad remains the soft ambient/fill source.
        push_sphere(
            point3(0.0, 200.0, 18.0),
            7.5,
            "sun_light",
            true
        );

        struct cloud_lobe {
            double x;
            double y;
            double z;
            double radius;
            double density_scale;
            bool lighter;
        };

        const cloud_lobe lobes[] = {
            // One long, connected, dense underside. Adjacent lobes overlap so
            // the bank reads as one storm front instead of separate spheres.
            {-12.0, -1.0,  0.0, 2.5, 0.85, false},
            { -9.5, -0.9, -0.8, 2.8, 1.00, false},
            { -6.8, -1.1,  0.6, 2.7, 1.15, false},
            { -4.0, -0.8, -0.4, 3.0, 1.05, false},
            { -1.0, -1.0,  0.8, 2.8, 1.20, false},
            {  1.8, -0.9, -0.7, 3.1, 1.10, false},
            {  4.8, -1.1,  0.4, 2.7, 1.20, false},
            {  7.5, -0.8, -0.5, 2.9, 0.95, false},
            { 10.3, -1.0,  0.7, 2.6, 1.10, false},
            { 12.7, -0.7, -0.2, 2.3, 0.75, false},

            // Uneven upper billows break up the silhouette and density.
            {-11.0,  0.8, -1.0, 2.1, 0.45, true},
            { -8.2,  1.5,  0.4, 2.5, 0.60, true},
            { -5.0,  0.9, -0.5, 2.2, 0.50, true},
            { -2.5,  2.0,  0.6, 2.8, 0.70, true},
            {  0.8,  1.1, -0.8, 2.3, 0.55, true},
            {  3.5,  2.3,  0.3, 2.9, 0.75, true},
            {  6.8,  1.0, -0.4, 2.2, 0.50, true},
            {  9.3,  1.7,  0.8, 2.5, 0.65, true},
            { 11.8,  0.7, -0.6, 1.9, 0.40, true},

            // A few depth-offset lobes make the volume less planar.
            { -7.0,  0.0,  2.5, 2.3, 0.65, false},
            { -1.5,  0.5, -2.4, 2.6, 0.55, true},
            {  4.5,  0.2,  2.7, 2.5, 0.70, false},
            {  9.0,  0.4, -2.2, 2.2, 0.50, true},
        };

        auto add_cloud = [&](double moving_x, double y, double z, double scale,
                             std::uint32_t variation_seed) {
            // Keep each bank deep enough to cast a substantial patch rather
            // than the thin line produced by the original planar layout.
            constexpr double width_scale = 0.55;
            constexpr double depth_scale = 1.55;
            point3 origin(moving_x, y, z);
            point3 noise_origin =
                origin - vec3(11.0 * cloud_random01(clouds.seed, variation_seed + 1u),
                              7.0 * cloud_random01(clouds.seed, variation_seed + 2u),
                              9.0 * cloud_random01(clouds.seed, variation_seed + 3u));
            heterogeneous_medium medium(
                point3(moving_x - 11.5 * scale, y - 3.2 * scale, z - 9.5 * scale),
                point3(moving_x + 11.5 * scale, y + 6.5 * scale, z + 9.5 * scale),
                noise_origin, clouds.density, y - 3.0 * scale,
                material_id_map.at("cloud_dark"), material_id_map.at("cloud_light"));
            for (int lobe_index = 0;
                 lobe_index < static_cast<int>(sizeof(lobes) / sizeof(lobes[0]));
                 ++lobe_index) {
                const auto& lobe = lobes[lobe_index];
                std::uint32_t lobe_stream = variation_seed + 20u + lobe_index * 4u;
                double radius_variation =
                    0.84 + 0.32 * cloud_random01(clouds.seed, lobe_stream);
                double y_variation =
                    (cloud_random01(clouds.seed, lobe_stream + 1u) - 0.5) * 0.7;
                double z_variation =
                    (cloud_random01(clouds.seed, lobe_stream + 2u) - 0.5) * 1.1;
                double density_variation =
                    0.85 + 0.30 * cloud_random01(clouds.seed, lobe_stream + 3u);
                medium.add_lobe(
                    point3(moving_x + scale * width_scale * lobe.x,
                           y + scale * (lobe.y + y_variation),
                           z + scale * depth_scale * (lobe.z + z_variation)),
                    scale * lobe.radius * radius_variation,
                    lobe.density_scale * density_variation, lobe.lighter);
            }
            host_objects.push_back({});
            host_objects.back().type = object_type::HETEROGENEOUS_MEDIUM;
            host_objects.back().heterogeneous_medium_data = medium;
        };

        // Populate the sky from the first frame. Each independently varied
        // bank travels toward screen-left, then wraps to screen-right with a
        // new size, height, depth, and noise pattern. Staggered phases make
        // new pieces enter continuously instead of exposing an empty sky.
        for (int bank = 0; bank < clouds.bank_count; ++bank) {
            double phase = (bank + 0.18 * cloud_random01(clouds.seed, 100u + bank)) /
                           clouds.bank_count;
            double unwrapped_x = clouds.start_x + phase * clouds.travel_span + clouds.speed * t;
            double cycles = std::floor((unwrapped_x - clouds.start_x) / clouds.travel_span);
            double moving_x = unwrapped_x - cycles * clouds.travel_span;
            int cycle_index = static_cast<int>(cycles);
            std::uint32_t variation_seed =
                10000u + static_cast<std::uint32_t>(bank) * 1000u +
                static_cast<std::uint32_t>(cycle_index + 100000) * 79u;

            double scale = 0.72 + 0.28 * cloud_random01(clouds.seed, variation_seed + 4u);
            // Keep the cloud base safely above the tallest tree canopy.
            double y = 11.5 + 1.0 * cloud_random01(clouds.seed, variation_seed + 5u);
            double z = 15.0 + 6.0 * cloud_random01(clouds.seed, variation_seed + 6u);
            add_cloud(moving_x, y, z, scale, variation_seed);
        }
    }

    // ------------------------------------------------
    // 9. Lightning flash and visible branching bolt
    // The small sphere is included in the light PDF and illuminates the scene.
    // Emissive cylinders draw the bolt but are not treated as sampled lights.
    // ------------------------------------------------
    if (lightning_frame.active) {
        double bolt_visual_scale = lightning_frame.event.visual_scale;
        push_sphere(lightning_frame.event.cloud_point, 0.45 * bolt_visual_scale,
                    "lightning_flash", true);

        constexpr int main_segment_count = 10;
        point3 main_points[main_segment_count + 1];
        point3 start = lightning_frame.event.cloud_point;
        point3 end = lightning_frame.event.ground_point;
        for (int segment = 0; segment <= main_segment_count; ++segment) {
            double fraction = segment / static_cast<double>(main_segment_count);
            point3 point = (1.0 - fraction) * start + fraction * end;
            if (segment > 0 && segment < main_segment_count) {
                double taper = std::sin(pi * fraction);
                double jitter_x =
                    lightning_random01(lightning.seed,
                                       3000u + lightning_frame.event.index * 97u + segment * 2u) -
                    0.5;
                double jitter_z =
                    lightning_random01(lightning.seed,
                                       3001u + lightning_frame.event.index * 97u + segment * 2u) -
                    0.5;
                point += vec3(jitter_x * 3.0 * taper, 0.0, jitter_z * 2.0 * taper);
            }
            main_points[segment] = point;
            if (segment > 0) {
                double radius = 0.055 * bolt_visual_scale * (1.0 - 0.45 * fraction);
                push_cylinder(main_points[segment - 1], main_points[segment], radius,
                              "lightning_bolt");
            }
        }

        const int branch_roots[] = {3, 6};
        for (int branch = 0; branch < 2; ++branch) {
            point3 branch_start = main_points[branch_roots[branch]];
            double direction = branch == 0 ? -1.0 : 1.0;
            point3 branch_end =
                branch_start + vec3(direction * (2.2 + branch), -2.5 - 0.6 * branch,
                                    (branch == 0 ? 1.4 : -1.1));
            point3 previous = branch_start;
            for (int segment = 1; segment <= 3; ++segment) {
                double fraction = segment / 3.0;
                point3 point = (1.0 - fraction) * branch_start + fraction * branch_end;
                if (segment < 3) {
                    double jitter =
                        lightning_random01(
                            lightning.seed, 5000u + lightning_frame.event.index * 31u +
                                                branch * 7u + segment) -
                        0.5;
                    point += vec3(jitter * 0.7, 0.0, -jitter * 0.5);
                }
                push_cylinder(previous, point, 0.025 * bolt_visual_scale, "lightning_bolt");
                previous = point;
            }
        }
    }

    // ------------------------------------------------
    // 10. Far mountains / silhouettes
    // Overlapping elliptical paraboloids form a continuous rounded range.
    // Their capped bases sit below the ground so only the curved sides show.
    // ------------------------------------------------
    const double mountain_base_y = ground_y - 0.35;

    // Back row: broad, lighter silhouettes.
    push_paraboloid(point3(-11.0, mountain_base_y, 45.0), 12.0, 9.0, 7.0,
                    "mountain_far");
    push_paraboloid(point3(  1.0, mountain_base_y, 47.0), 14.0, 10.0, 10.0,
                    "mountain_far");
    push_paraboloid(point3( 13.0, mountain_base_y, 45.0), 10.0, 9.0, 7.5,
                    "mountain_far");

    // Front row: darker forms overlap the back row and give the range depth.
    push_paraboloid(point3(-8.0, mountain_base_y, 37.0), 9.0, 8.0, 5.5,
                    "mountain_mid");
    push_paraboloid(point3( 8.5, mountain_base_y, 38.0), 10.0, 8.0, 6.2,
                    "mountain_near");

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
        /*image_width=*/quality.image_width,
        /*samples_per_pixel=*/quality.samples_per_pixel,
        /*max_depth=*/quality.max_depth,
        /*aspect_ratio=*/16.0 / 9.0,
        /*vfov=*/40.0,
        /*lookfrom=*/point3(0.0, 4.2, -2.5),
        /*lookat=*/point3(0.0, 4.0, 16.0),
        /*vup=*/vec3(0.0, 1.0, 0.0),
        /*defocus_angle=*/0.0,
        /*focus_dist=*/18.0,
        // /*background=*/color(0.04, 0.05, 0.09)
        /*background=*/clouds.enabled ? color(0.28, 0.32, 0.38)
                                      : color(0.68, 0.73, 0.80)
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
        rain,
        frame
    );
}
