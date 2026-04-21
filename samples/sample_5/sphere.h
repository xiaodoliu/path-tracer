#pragma once

#include "hittable.h"
#include <cassert>

class sphere : public hittable{
public:
    HD sphere() = default;
    HD sphere(const point3& center, double radius): center(center), radius(radius){
        assert(radius > 0);
    }
    D bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override{
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(oc, r.direction());
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = h * h - a * c;
        if(discriminant < 0.0){
            return false;
        }
        auto sqrtd = std::sqrt(discriminant);
        auto t = (h - sqrtd) / a;
        if(t > ray_tmax){
            return false;
        }
        if(t < ray_tmin){
            t = (h + sqrtd) / a;
            if(t < ray_tmin || t > ray_tmax){
                return false;
            }
        }
        rec.t = t;
        rec.p = r.at(t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }

private:
    point3 center;
    double radius;
};
