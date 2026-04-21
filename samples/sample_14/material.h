#pragma once

#include "hittable.h"

enum class material_type{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
};

struct material_data{
    material_type type;
    color albedo;
    double fuzz = 0.0; // for metal material
    double refraction_index = 1.0; // for dielectric material
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
        case material_type::DIELECTRIC:{
            attenuation = color(1.0, 1.0, 1.0);
            double ri = rec.front_face ? 1.0/mat_data.refraction_index : mat_data.refraction_index;
            vec3 unit_direction = normalize(r_in.direction());
            double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);
            bool cannot_refract = ri * sin_theta > 1.0;
            vec3 direction;
            if(cannot_refract){
                direction = reflect(unit_direction, rec.normal);
            }else{
                direction = refract(unit_direction, rec.normal, ri);
            }
            scattered = ray(rec.p, direction);
            return true;
        }
        default:
            return false;
    }
}
