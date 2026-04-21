# pragma once

#include "vec3.h"

class ray{
public:
    HD ray() = default;
    HD ray(const point3& orig, const vec3& d) : orig(orig), d(d){}

    HD const point3 origin() const {return orig;}
    HD const vec3 direction() const {return d;}
    HD point3 at(double t) const {return orig + t * d;}
    
private:
    point3 orig;
    vec3 d;
};