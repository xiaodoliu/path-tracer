#pragma once

#include "hittable.h"
#include "texture.h"
#include "onb.h"

enum class material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    ISOTROPIC,
    HENYEY_GREENSTEIN,
};
enum class scatter_pdf_type {
    NONE,
    COSINE,
    SPHERE,
    HENYEY_GREENSTEIN,
};
struct scatter_record {
    color attenuation;
    scatter_pdf_type pdf_type = scatter_pdf_type::NONE;
    bool skip_pdf = false;
    ray skip_pdf_ray;
    vec3 pdf_axis;
    double anisotropy = 0.0;
};

struct material_data {
    material_type type;
    color albedo;
    int texture_id = -1;
    double fuzz = 0.0;              // for metal material
    double refraction_index = 1.0;  // for dielectric material
    bool receives_rain_post_process = false;
    double anisotropy = 0.0;        // Henyey-Greenstein g (-1 back, +1 forward)
};

enum rain_mode {
    NONE,
    POST_PROCESSING,
    PATH_TRACING,
    HYBRID,
};

struct rain_settings {
    rain_mode mode = rain_mode::POST_PROCESSING;
    float size = 3.7f;
    float time = 0.0f;
    float time_offset = 1.0f;  // Unity's _T property
    float distortion = -5.0f;
    float blur = 0.05f;
    int frame_count = 1;
    float frames_per_second = 30.0f;
    float time_scale = 1.0f;
};

D inline double reflectance(double cosine, double ri) {
    auto r0 = (1 - ri) / (1 + ri);
    r0 *= r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}

D inline double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered,
                               const material_data& mat_data) {
    switch (mat_data.type) {
        case material_type::LAMBERTIAN: {
            auto cos_theta = dot(rec.normal, normalize(scattered.direction()));
            return cos_theta < 0 ? 0 : cos_theta / pi;
        }
        case material_type::ISOTROPIC: {
            return 1 / (4 * pi);
        }
        case material_type::HENYEY_GREENSTEIN: {
            double g = mat_data.anisotropy;
            double cosine = dot(normalize(r_in.direction()), normalize(scattered.direction()));
            double denominator = 1.0 + g * g - 2.0 * g * cosine;
            return (1.0 - g * g) / (4.0 * pi * denominator * sqrt(denominator));
        }
        default:
            return 0;
    }
}

D inline bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec,
                      const texture_data* device_textures, const material_data& mat_data,
                      curandState* state) {
    switch (mat_data.type) {
        case material_type::LAMBERTIAN: {
            srec.attenuation =
                mat_data.texture_id >= 0
                    ? texture_value(rec.u, rec.v, rec.p, device_textures[mat_data.texture_id])
                    : mat_data.albedo;
            srec.pdf_type = scatter_pdf_type::COSINE;
            srec.pdf_axis = rec.normal;
            srec.anisotropy = 0.0;
            srec.skip_pdf = false;
            return true;
        }
        case material_type::METAL: {
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = normalize(reflected) + mat_data.fuzz * random_unit_vector(state);
            srec.attenuation = mat_data.albedo;
            srec.skip_pdf = true;
            srec.skip_pdf_ray = ray(rec.p, reflected, r_in.time());
            return true;
        }
        case material_type::DIELECTRIC: {
            double ri =
                rec.front_face ? 1.0 / mat_data.refraction_index : mat_data.refraction_index;
            vec3 unit_direction = normalize(r_in.direction());
            double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
            bool cannot_refract = ri * sin_theta > 1.0;
            vec3 direction;
            if (cannot_refract || reflectance(cos_theta, ri) > random_double(state)) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, ri);
            }
            srec.attenuation = color(1.0, 1.0, 1.0);
            srec.skip_pdf = true;
            srec.skip_pdf_ray = ray(rec.p, direction, r_in.time());
            return true;
        }
        case material_type::ISOTROPIC: {
            srec.attenuation =
                mat_data.texture_id >= 0
                    ? texture_value(rec.u, rec.v, rec.p, device_textures[mat_data.texture_id])
                    : mat_data.albedo;
            srec.pdf_type = scatter_pdf_type::SPHERE;
            srec.pdf_axis = rec.normal;
            srec.anisotropy = 0.0;
            srec.skip_pdf = false;
            return true;
        }
        case material_type::HENYEY_GREENSTEIN: {
            srec.attenuation =
                mat_data.texture_id >= 0
                    ? texture_value(rec.u, rec.v, rec.p, device_textures[mat_data.texture_id])
                    : mat_data.albedo;
            srec.pdf_type = scatter_pdf_type::HENYEY_GREENSTEIN;
            srec.pdf_axis = normalize(r_in.direction());
            srec.anisotropy = mat_data.anisotropy;
            srec.skip_pdf = false;
            return true;
        }
        default:
            return false;
    }
}

D inline color emitted(const ray& r_in, const hit_record& rec, const texture_data* device_textures,
                       const material_data& mat_data) {
    switch (mat_data.type) {
        case material_type::DIFFUSE_LIGHT: {
            if (!rec.front_face) {
                return color(0.0, 0.0, 0.0);
            }
            return mat_data.texture_id >= 0
                       ? texture_value(rec.u, rec.v, rec.p, device_textures[mat_data.texture_id])
                       : mat_data.albedo;
        }
        default:
            return color(0.0, 0.0, 0.0);
    }
}
