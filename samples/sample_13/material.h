#pragma once

#include "hittable.h"

enum class material_type{
    LAMBERTIAN,
    METAL,
};

struct material_data{
    material_type type;
    color albedo;
    double fuzz = 0.0; // for metal material
};

D inline bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, const material_data& mat_data, curandState* state){
    switch(mat_data.type){
        case material_type::LAMBERTIAN:{
            auto scatter_direction = rec.normal + random_unit_vector(state);
            if(scatter_direction.near_zero()){
                scatter_direction = rec.normal;
            }
            scattered = ray(rec.p, scatter_direction);
            attenuation = mat_data.albedo;
            return true;
        }
        case material_type::METAL:{
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = normalize(reflected) + mat_data.fuzz * random_unit_vector(state);
            scattered = ray(rec.p, reflected);
            attenuation = mat_data.albedo;
            return dot(scattered.direction(), rec.normal) > 0;
        }
        default:
            return false;
    }
}
