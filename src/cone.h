#pragma once

#include "hittable.h"

class cone {
   public:
    H cone() = default;
    H cone(const point3& apex, const point3& base_center, double radius, int material_id)
        : apex(apex),
          base_center(base_center),
          radius(radius),
          height((apex - base_center).length()),
          material_id(material_id) {
        assert(radius > 0);
        assert(height > 0);
        vec3 axis = normalize(apex - base_center);
        double perpendicular_x = std::sqrt(std::fmax(1.0 - axis.x() * axis.x(), 0.0));
        double perpendicular_y = std::sqrt(std::fmax(1.0 - axis.y() * axis.y(), 0.0));
        double perpendicular_z = std::sqrt(std::fmax(1.0 - axis.z() * axis.z(), 0.0));
        vec3 radius_extent = radius * vec3(perpendicular_x, perpendicular_y, perpendicular_z);
        point3 base_max = base_center + radius_extent;
        point3 base_min = base_center - radius_extent;
        point3 max(std::fmax(apex.x(), base_max.x()), std::fmax(apex.y(), base_max.y()),
                   std::fmax(apex.z(), base_max.z()));
        point3 min(std::fmin(apex.x(), base_min.x()), std::fmin(apex.y(), base_min.y()),
                   std::fmin(apex.z(), base_min.z()));
        bbox = aabb(min, max);
    }
    HD aabb bounding_box() const { return bbox; }
    D bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        const double eps = 1e-12;
        double closest_so_far = ray_t.max;
        bool hit_anything = false;
        // Check if the ray hits the cone sides.
        vec3 AO = r.origin() - apex;
        vec3 Dir = r.direction();
        vec3 n = normalize(base_center - apex);
        double k = radius / height;
        double dot_AO_n = dot(AO, n);
        double dot_AO_D = dot(AO, Dir);
        double dot_D_n = dot(Dir, n);
        double a = Dir.length_squared() - (1 + k * k) * dot_D_n * dot_D_n;
        double half_b = dot_AO_D - (1 + k * k) * dot_AO_n * dot_D_n;
        double c = AO.length_squared() - (1 + k * k) * dot_AO_n * dot_AO_n;
        double discriminant = half_b * half_b - a * c;
        if (discriminant >= -eps && std::fabs(a) > eps) {
            double sqrtd = std::sqrt(fmax(0.0, discriminant));
            double t = (-half_b - sqrtd) / a;
            vec3 AP = r.at(t) - apex;
            double dot_AP_n = dot(AP, n);
            if (dot_AP_n > eps && dot_AP_n <= height + eps) {
                if (t >= ray_t.min && t <= closest_so_far) {
                    vec3 outward_normal = normalize(AP - (1 + k * k) * dot_AP_n * n);
                    rec.t = t;
                    rec.p = r.at(t);
                    rec.material_id = material_id;
                    rec.set_face_normal(r, outward_normal);
                    hit_anything = true;
                    closest_so_far = t;
                }
            }
            t = (-half_b + sqrtd) / a;
            AP = r.at(t) - apex;
            dot_AP_n = dot(AP, n);
            if (dot_AP_n > eps && dot_AP_n <= height + eps) {
                if (t >= ray_t.min && t <= closest_so_far) {
                    vec3 outward_normal = normalize(AP - (1 + k * k) * dot_AP_n * n);
                    rec.t = t;
                    rec.p = r.at(t);
                    rec.material_id = material_id;
                    rec.set_face_normal(r, outward_normal);
                    hit_anything = true;
                    closest_so_far = t;
                }
            }
        } else if (std::fabs(a) < eps && std::fabs(half_b) > eps) {
            double t = -c / (2 * half_b);
            vec3 AP = r.at(t) - apex;
            double dot_AP_n = dot(AP, n);
            if (dot_AP_n > eps && dot_AP_n <= height + eps) {
                if (t >= ray_t.min && t <= closest_so_far) {
                    vec3 outward_normal = normalize(AP - (1 + k * k) * dot_AP_n * n);
                    rec.t = t;
                    rec.p = r.at(t);
                    rec.material_id = material_id;
                    rec.set_face_normal(r, outward_normal);
                    hit_anything = true;
                    closest_so_far = t;
                }
            }
        }

        // base cap
        vec3 OC0 = base_center - r.origin();
        double dot_n_OC0 = dot(n, OC0);
        if (std::fabs(dot_D_n) > eps) {
            double t = dot_n_OC0 / dot_D_n;
            vec3 C0P = r.at(t) - base_center;
            vec3 radial = C0P - dot(C0P, n) * n;
            if (t >= ray_t.min && t <= closest_so_far &&
                radial.length_squared() <= radius * radius + eps) {
                rec.t = t;
                rec.p = r.at(t);
                rec.material_id = material_id;
                rec.set_face_normal(r, n);
                hit_anything = true;
                closest_so_far = t;
            }
        }

        return hit_anything;
    }

   private:
    point3 apex, base_center;
    double radius, height;
    int material_id;
    aabb bbox;
};