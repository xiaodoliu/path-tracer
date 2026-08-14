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
