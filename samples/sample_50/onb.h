#pragma once

#include "vec3.h"

class onb {
public:
    HD onb(const vec3& n){
        axis[2] = normalize(n);
        vec3 a = (std::fabs(axis[2].x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        axis[1] = normalize(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }

    HD const vec3& u() const {return axis[0];}
    HD const vec3& v() const {return axis[1];}
    HD const vec3& w() const {return axis[2];}

    HD vec3 transform(const vec3& v) const {
        return v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    }

private:
    vec3 axis[3];
};