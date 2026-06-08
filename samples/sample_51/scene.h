#pragma once

#include "ray.h"
#include "vec3.h"
#include "hittable.h"
#include "sphere.h"
#include "quad.h"
#include "box.h"
#include "constant_medium.h"
#include "interval.h"
#include "bvh.h"

enum class object_type{
    SPHERE,
    QUAD,
    BOX,
    CONSTANT_MEDIUM,
};

struct scene_object{
    object_type type;

    sphere sphere_data;
    quad quad_data;
    box box_data;
    constant_medium constant_medium_data;
};

inline void add_quad_transformed(
    scene_object* objects,
    int& object_index,
    const point3& Q,
    const vec3& u,
    const vec3& v,
    int meterial_id,
    double angle,
    const vec3& offset
){
    objects[object_index].type = object_type::QUAD;
    objects[object_index++].quad_data = quad(
        rotate_y(Q, angle) + offset,
        rotate_y(u, angle),
        rotate_y(v, angle),
        meterial_id
    );
}

D inline bool hit(const ray& r, interval ray_t, hit_record& rec, const scene_object& object, curandState* state = nullptr){
    switch(object.type){
        case object_type::SPHERE:
            return object.sphere_data.hit(r, ray_t, rec);
        case object_type::QUAD:
            return object.quad_data.hit(r, ray_t, rec);
        case object_type::BOX:
            return object.box_data.hit(r, ray_t, rec);
        case object_type::CONSTANT_MEDIUM:
            return object.constant_medium_data.hit(r, ray_t, rec, state);
        default:
            return false;
    }
}

HD aabb bounding_box(const scene_object& object){
    switch(object.type){
        case object_type::SPHERE:
            return object.sphere_data.bounding_box();
        case object_type::QUAD:
            return object.quad_data.bounding_box();
        case object_type::BOX:
            return object.box_data.bounding_box();
        case object_type::CONSTANT_MEDIUM:
            return object.constant_medium_data.bounding_box();
        default:
            return aabb(point3(0, 0, 0), point3(0, 0, 0));
    }
}

D bool hit_bvh(const ray& r, 
    interval ray_t,
    hit_record& rec,
    const bvh_node* nodes,
    const int* prim_indices,
    const scene_object* objects,
    curandState* state = nullptr){

int stack[64];
int sp = 0;
stack[sp++] = 0;
bool hit_anything = false;
double closest = ray_t.max;

while(sp > 0){
    int node_index = stack[--sp];
    const bvh_node& node = nodes[node_index];
    if(!node.box.hit(r, interval(ray_t.min, closest))) {
        continue;
    }
    bool is_leaf = (node.left == -1 && node.right == -1);
    if(is_leaf){
        for(int i = 0; i < node.count; ++i){
            int obj_idx = prim_indices[node.start + i];
            if(hit(r, interval(ray_t.min, closest), rec, objects[obj_idx], state)){
                hit_anything = true;
                closest = rec.t;
            }
        }
    }else{
        stack[sp++] = node.right;
        stack[sp++] = node.left;
    }
}

return hit_anything;
}

int build_bvh(std::vector<bvh_node>& nodes,
    std::vector<int>& prim_indices,
    const scene_object* objects,
    int start,
    int end){
    int node_index = static_cast<int>(nodes.size());
    nodes.push_back({});

    int count = end-start;
    aabb node_box = bounding_box(objects[prim_indices[start]]);
    for(int i = start+1; i < end; ++i){
        node_box = aabb(node_box, bounding_box(objects[prim_indices[i]]));
    }
    const int leaf_node_threshold = 2;
    if(count <= leaf_node_threshold){
    nodes[node_index] = bvh_node{node_box, -1, -1, start, count};
    return node_index;
}

int axis = node_box.longest_axis();

std::sort(prim_indices.begin() + start,
      prim_indices.begin() + end,
      [&](int a, int b){
        aabb box_a = bounding_box(objects[a]);
        aabb box_b = bounding_box(objects[b]);
        double ca = box_a.axis_interval(axis).min + box_a.axis_interval(axis).max;
        double cb = box_b.axis_interval(axis).min + box_b.axis_interval(axis).max;
        return ca < cb;
      } 
    );
int mid = start + count / 2;
int left = build_bvh(nodes,prim_indices, objects,  start, mid);
int right = build_bvh(nodes,prim_indices, objects, mid, end);
nodes[node_index] = bvh_node{node_box, left, right, 0, 0};
return node_index;
}

