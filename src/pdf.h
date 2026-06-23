#pragma once

#include "onb.h"
#include "hittable.h"

D inline double sphere_pdf_value(const vec3& direction){
    return 1/(4*pi);
}

D inline vec3 sphere_pdf_generate(curandState* state){
    return random_unit_vector(state);
}

D inline double cosine_pdf_value(const vec3& direction, const vec3& normal){
    auto cosine_theta = dot(normalize(direction), normal);
    return cosine_theta < 0 ? 0 : cosine_theta / pi;
}

D inline vec3 cosine_pdf_generate(const vec3& normal, curandState* state){
    onb uvw(normal);
    return uvw.transform(random_cosine_direction(state));
}

D inline double object_pdf_value(const point3& origin, const vec3& direction, const scene_object& object){
    switch(object.type){
        case object_type::QUAD:
            return object.quad_data.pdf_value(origin, direction);
        case object_type::SPHERE:
            return object.sphere_data.pdf_value(origin, direction);
        default:
            return 0;
    }
}

D inline vec3 object_pdf_generate(const point3& origin, const scene_object& object, curandState* state){
    switch(object.type){
        case object_type::QUAD:
            return object.quad_data.random(origin, state);
        case object_type::SPHERE:
            return object.sphere_data.random(origin, state);
        default:
            return vec3(0, 0, 0);
    }
}

D inline double light_list_pdf_value(const point3& origin, const vec3& direction, const scene_object* objects, 
    const int* light_indices, int num_lights){
    if(num_lights == 0){
        return 0.0;
    }
    double sum = 0.0;
    double weight = 1.0 / num_lights;
    for(int i = 0; i < num_lights; ++i){
        int object_index = light_indices[i];
        sum += weight * object_pdf_value(origin, direction, objects[object_index]);
    }
    return sum;
}

D inline vec3 light_list_pdf_generate(const point3& origin, const scene_object* objects, 
    const int* light_indices, int num_lights, curandState* state) {
        int i = static_cast<int>(random_double(state) * num_lights);
        if(i >= num_lights) {
            i = num_lights - 1;
        }
        int object_index = light_indices[i];
        return object_pdf_generate(origin, objects[object_index], state);
    }

D inline double material_pdf_value(const vec3& direction, const vec3& normal, scatter_pdf_type type){
    switch(type){
        case scatter_pdf_type::COSINE: return cosine_pdf_value(direction, normal);
        case scatter_pdf_type::SPHERE: return sphere_pdf_value(direction);
        default: return 0;
    }
}

D inline vec3 material_pdf_generate(const vec3& normal, scatter_pdf_type type, curandState* state){
    switch(type){
        case scatter_pdf_type::COSINE: return cosine_pdf_generate(normal, state);
        case scatter_pdf_type::SPHERE: return sphere_pdf_generate(state);
        default: return vec3(0, 0, 0);
    }
}

D inline double mixture_pdf_value(const point3& origin, const vec3& direction, const vec3& normal, 
    const scene_object* objects, const int* light_indices, int num_lights, scatter_pdf_type material_pdf_type){
    double p1 = material_pdf_value(direction, normal, material_pdf_type);
    if(num_lights == 0) return p1;
    double p0 = light_list_pdf_value(origin, direction, objects, light_indices, num_lights);
    return 0.5 * p0 + 0.5 * p1;
}

D inline vec3 mixture_pdf_generate(const point3& origin, const vec3& normal, 
    const scene_object* objects, const int* light_indices, int num_lights, scatter_pdf_type material_pdf_type, curandState* state) {
    if(num_lights > 0 && random_double(state) < 0.5) {
        return light_list_pdf_generate(origin, objects, light_indices, num_lights, state);
    }
    return material_pdf_generate(normal, material_pdf_type, state);
}