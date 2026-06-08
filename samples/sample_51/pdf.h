#pragma once

#include "onb.h"

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
