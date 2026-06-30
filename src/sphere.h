#pragma once

#include "hittable.h"
#include "ray.h"
#include "onb.h"
#include <cassert>

class sphere {
   public:
    HD sphere() = default;
    HD sphere(const point3& center, double radius, int material_id)
        : center(center, vec3(0, 0, 0)), radius(radius), material_id(material_id) {
        assert(radius > 0);
        vec3 rvec = vec3(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }
    HD sphere(const point3& center1, const point3& center2, double radius, int material_id)
        : center(center1, center2 - center1), radius(radius), material_id(material_id) {
        assert(radius > 0);
        vec3 rvec = vec3(radius, radius, radius);
        aabb box1(center1 - rvec, center1 + rvec);
        aabb box2(center2 - rvec, center2 + rvec);
        bbox = aabb(box1, box2);
    }
    D bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        auto current_center = center.at(r.time());
        vec3 oc = current_center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(oc, r.direction());
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = h * h - a * c;
        if (discriminant < 0.0) {
            return false;
        }
        auto sqrtd = std::sqrt(discriminant);
        auto t = (h - sqrtd) / a;
        if (ray_t.less_than(t)) {
            return false;
        }
        if (ray_t.greater_than(t)) {
            t = (h + sqrtd) / a;
            if (!ray_t.contains(t)) {
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

    D double pdf_value(const point3& origin, const vec3& direction) const {
        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001, infinity), rec)) {
            return 0;
        }
        auto dist_squared = (center.at(0) - origin).length_squared();
        auto cos_theta_max = std::sqrt(1 - radius * radius / dist_squared);
        auto solid_angle = 2 * pi * (1 - cos_theta_max);
        return 1 / solid_angle;
    }

    D vec3 random(const point3& origin, curandState* state) const {
        vec3 direction = center.at(0) - origin;
        auto distance_squared = direction.length_squared();
        onb uvw(direction);
        return uvw.transform(random_to_sphere(radius, distance_squared, state));
    }

   private:
    ray center;
    double radius;
    int material_id;
    aabb bbox;

    D static void get_sphere_uv(const point3& p, double& u, double& v) {
        auto theta = std::acos(-p.y());
        auto phi = std::atan2(-p.z(), p.x()) + pi;
        u = phi / (2 * pi);
        v = theta / pi;
    }

    D static vec3 random_to_sphere(double radius, double distance_squared, curandState* state) {
        auto r1 = random_double(state);
        auto r2 = random_double(state);
        auto z = 1 - r2 * (1 - std::sqrt(1 - radius * radius / distance_squared));
        auto phi = 2 * pi * r1;
        auto x = std::cos(phi) * std::sqrt(1 - z * z);
        auto y = std::sin(phi) * std::sqrt(1 - z * z);
        return vec3(x, y, z);
    }
};
