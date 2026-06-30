#pragma once

#include "vec3.h"

class ray {
   public:
    HD ray() = default;
    HD ray(const point3& orig, const vec3& d, double tm) : orig(orig), d(d), tm(tm) {}
    HD ray(const point3& orig, const vec3& d) : ray(orig, d, 0.0) {}

    HD const point3 origin() const { return orig; }
    HD const vec3 direction() const { return d; }
    HD double time() const { return tm; }
    HD point3 at(double t) const { return orig + t * d; }

   private:
    point3 orig;
    vec3 d;
    double tm;
};