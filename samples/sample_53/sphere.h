#pragma once

#include "hittable.h"
#include "ray.h"
#include <cassert>

class sphere{
public:
    HD sphere() = default;
    HD sphere(const point3& center, double radius, int material_id): center(center, vec3(0, 0, 0)), radius(radius), material_id(material_id){
        assert(radius > 0);
        vec3 rvec = vec3(radius, radius, radius);
        bbox = aabb(center-rvec, center+rvec);
    }
    HD sphere(const point3& center1, const point3& center2, double radius, int material_id): center(center1, center2 - center1), radius(radius), material_id(material_id){
        assert(radius > 0);
        vec3 rvec = vec3(radius, radius, radius);
        aabb box1(center1-rvec, center1+rvec);
        aabb box2(center2-rvec, center2+rvec);
        bbox = aabb(box1, box2);
    }
    D bool hit(const ray& r, interval ray_t, hit_record& rec) const{
        auto current_center = center.at(r.time());
        vec3 oc = current_center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(oc, r.direction());
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = h * h - a * c;
        if(discriminant < 0.0){
            return false;
        }
        auto sqrtd = std::sqrt(discriminant);
        auto t = (h - sqrtd) / a;
        if(ray_t.less_than(t)){
            return false;
        }
        if(ray_t.greater_than(t)){
            t = (h + sqrtd) / a;
            if(!ray_t.contains(t)){
                return false;
            }
        }
        rec.t = t;
        rec.p = r.at(t);
        vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.material_id = material_id;
        return true;
    }

    HD aabb bounding_box() const { return bbox; }
private:
    ray center;
    double radius;
    int material_id;
    aabb bbox;

    D static void get_sphere_uv(const point3& p, double& u, double& v){
        auto theta = std::acos(-p.y());
        auto phi = std::atan2(-p.z(), p.x()) + pi;
        u = phi / (2*pi);
        v = theta / pi;
    }
};
