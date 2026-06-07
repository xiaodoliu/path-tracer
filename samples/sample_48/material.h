#pragma once

#include "hittable.h"
#include "texture.h"
#include "onb.h"

enum class material_type{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    ISOTROPIC,
};

struct material_data{
    material_type type;
    color albedo;
    int texture_id = -1;
    double fuzz = 0.0; // for metal material
    double refraction_index = 1.0; // for dielectric material
};

D inline double reflectance(double cosine, double ri){
    auto r0 = (1-ri) / (1+ri);
    r0 *= r0;
    return r0 + (1-r0)*std::pow(1-cosine, 5);
}

D inline double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered, const material_data& mat_data){
    switch(mat_data.type){
        case material_type::LAMBERTIAN:{
            return 1 / (2 * pi);
        }
        case material_type::ISOTROPIC:{
            return 1 / (4 * pi);
        }
        default:
            return 0;
    }
}

D inline bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, const texture_data* device_textures, const material_data& mat_data, double& pdf, curandState* state){
    switch(mat_data.type){
        case material_type::LAMBERTIAN:{
            onb uvw(rec.normal);
            auto scatter_direction = uvw.transform(random_cosine_direction(state));
            scattered = ray(rec.p, scatter_direction, r_in.time());
            attenuation = mat_data.texture_id >= 0 ? texture_value(rec.u, rec.v, rec.p, device_textures[mat_data.texture_id]) : mat_data.albedo;
            pdf = dot(uvw.w(), scatter_direction) / pi;
            return true;
        }
        case material_type::METAL:{
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = normalize(reflected) + mat_data.fuzz * random_unit_vector(state);
            scattered = ray(rec.p, reflected, r_in.time());
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
            if(cannot_refract || reflectance(cos_theta, ri) > random_double(state)){
                direction = reflect(unit_direction, rec.normal);
            }else{
                direction = refract(unit_direction, rec.normal, ri);
            }
            scattered = ray(rec.p, direction, r_in.time());
            return true;
        }
        case material_type::ISOTROPIC:{
            scattered = ray(rec.p, random_unit_vector(state), r_in.time());
            attenuation = mat_data.texture_id >= 0 ? 
                texture_value(rec.u, rec.v, rec.p, device_textures[mat_data.texture_id]) : 
                mat_data.albedo; 
            pdf = 1 / (4 * pi);
            return true;
        }
        default:
            return false;
    }
}

D inline color emitted(double u, double v, const point3& p, const texture_data* device_textures, const material_data& mat_data){
    switch(mat_data.type){
        case material_type::DIFFUSE_LIGHT:{
            return mat_data.texture_id >= 0 ? 
                texture_value(u ,v ,p, device_textures[mat_data.texture_id]) : 
                mat_data.albedo;
        }
        default:
            return color(0.0, 0.0, 0.0);
    }
}
