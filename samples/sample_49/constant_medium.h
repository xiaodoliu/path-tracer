# pragma once

#include "material.h"
#include "box.h"

struct constant_medium{
    enum class boundary_type{
        BOX,
        SPHERE,
    };
    boundary_type type;
    sphere sphere_boundary;
    box box_boundary;
    
    double neg_inv_density;
    int phase_function_material_id;
    aabb bbox;

    HD constant_medium() = default;
    HD constant_medium(const box& boundary, double density, int phase_function_material_id)
      : type(boundary_type::BOX), box_boundary(boundary), neg_inv_density(-1/density), 
      phase_function_material_id(phase_function_material_id),
      bbox(boundary.bounding_box()) {}
    HD constant_medium(const sphere& boundary, double density, int phase_function_material_id)
      : type(boundary_type::SPHERE), sphere_boundary(boundary), neg_inv_density(-1/density), 
      phase_function_material_id(phase_function_material_id),
      bbox(boundary.bounding_box()) {}
    
    D bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const{
        hit_record rec1, rec2;
        switch(type){
            case boundary_type::BOX:
                if(!box_boundary.hit(r, interval::universe(), rec1)){
                    return false;
                }
                if(!box_boundary.hit(r, interval(rec1.t+0.0001, infinity), rec2)){
                    return false;
                }
                break;
            case boundary_type::SPHERE:
                if(!sphere_boundary.hit(r, interval::universe(), rec1)){
                    return false;
                }
                if(!sphere_boundary.hit(r, interval(rec1.t+0.0001, infinity), rec2)){
                    return false;
                }
                break;
            default:
                return false;
        }
        if(rec1.t < ray_t.min) rec1.t = ray_t.min;
        if(rec2.t > ray_t.max) rec2.t = ray_t.max;

        if(rec1.t >= rec2.t) return false;
        if(rec1.t < 0) rec1.t = 0;

        auto ray_length = r.direction().length();
        auto distance_inside_boundary = (rec2.p - rec1.p). length();
        auto hit_distance = neg_inv_density * std::log(random_double(state));
        if(hit_distance > distance_inside_boundary) return false;
        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);
        rec.normal = vec3(1, 0, 0);
        rec.front_face = true;
        rec.material_id = phase_function_material_id;
        return true;
    }

    HD aabb bounding_box() const{
        return bbox;
    }  
};