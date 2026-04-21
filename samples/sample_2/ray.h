# pragma once

#include "vec3.h"

class ray{
public:
    HD ray() = default;
    HD ray(const point3& origin, const vec3& direction) : origin(origin), direction(direction){}

    HD const point3 origin() const {return origin;}
    HD const vec3 direction() const {return direction;}
    HD point3 at(double t) const {return origin + t * direction;}
    
private:
    point3 origin;
    vec3 direction;
};