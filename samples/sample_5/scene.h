#pragma once

#include "ray.h"
#include "vec3.h"
#include "hittable.h"

enum class object_type{
    SPHERE,
    QUAD,
};

struct scene_object{
    object_type type;

    sphere sphere_data;
    // quad quad_data;
};

D inline bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec, const scene_object& object){
    switch(object.type){
        case object_type::SPHERE:
            return object.sphere_data.hit(r, ray_tmin, ray_tmax, rec);
        case object_type::QUAD:
            return false;
        default:
            return false;
    }
}
