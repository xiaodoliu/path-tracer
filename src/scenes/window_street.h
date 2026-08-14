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
