#pragma once

#include "ray.h"
#include "vec3.h"
#include "interval.h"
#include "aabb.h"

class hit_record{
public:
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    int material_id;
    D void set_face_normal(const ray& r, const vec3& outward_normal){
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face? outward_normal : -outward_normal;
        normal = normalize(normal); // always normalize the normal vector
    }
};

class hittable{
public:
    virtual ~hittable() = default;
    D virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
    HD virtual aabb bounding_box() const = 0;
};