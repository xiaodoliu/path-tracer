# pragma once

#include "vec3.h"

struct camera{
    point3 origin = point3(0, 0, 0);
    double focal_length = 1.0;
    point3 pixel_sample_start;
    vec3 viewport_pixel_u_delta;
    vec3 viewport_pixel_v_delta;
};