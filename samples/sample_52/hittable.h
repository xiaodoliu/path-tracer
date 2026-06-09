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
    double u;
    double v;
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
    D virtual double pdf_value(const point3& origin, const vec3& direction) const {
        return 0.0;
    }
    D virtual vec3 random(const point3& origin, curandState* state = nullptr) const {
        return vec3(1, 0, 0);
    }
};