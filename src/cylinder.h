#pragma once

#include "hittable.h"

class cylinder {
   public:
    H cylinder() = default;
    H cylinder(const point3& top_center, const point3& base_center, double radius, int material_id)
        : top_center(top_center),
          base_center(base_center),
          height((top_center - base_center).length()),
          radius(radius),
          material_id(material_id) {
        assert(radius > 0);
        vec3 axis = normalize(top_center - base_center);
        double perpendicular_x = std::sqrt(std::fmax(1.0 - axis.x() * axis.x(), 0.0));
        double perpendicular_y = std::sqrt(std::fmax(1.0 - axis.y() * axis.y(), 0.0));
        double perpendicular_z = std::sqrt(std::fmax(1.0 - axis.z() * axis.z(), 0.0));
        point3 max = point3(std::fmax(top_center.x(), base_center.x()),
                            std::fmax(top_center.y(), base_center.y()),
                            std::fmax(top_center.z(), base_center.z())) +
                     radius * point3(perpendicular_x, perpendicular_y, perpendicular_z);
        point3 min = point3(std::fmin(top_center.x(), base_center.x()),
                            std::fmin(top_center.y(), base_center.y()),
                            std::fmin(top_center.z(), base_center.z())) -
                     radius * point3(perpendicular_x, perpendicular_y, perpendicular_z);
        bbox = aabb(min, max);
    }

    HD aabb bounding_box() const { return bbox; }

    D bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        const double eps = 1e-12;
        // Check if the ray hits the cylinder sides.
        double dot_D_D = r.direction().length_squared();
        // C0 - base center, C1 - top center,
        vec3 C0C1 = top_center - base_center;
        vec3 C0O = r.origin() - base_center;
        double dot_C0C1_C0C1 = C0C1.length_squared();
        // cylinder should have non-zero height
        if (dot_C0C1_C0C1 < eps) return false;
        double height_eps = 1e-9 * dot_C0C1_C0C1;
        double dot_C0C1_D = dot(C0C1, r.direction());
        double dot_C0O_D = dot(C0O, r.direction());
        double dot_C0C1_C0O = dot(C0C1, C0O);
        double dot_C0O_C0O = C0O.length_squared();
        double a = dot_D_D * dot_C0C1_C0C1 - dot_C0C1_D * dot_C0C1_D;
        double half_b = dot_C0O_D * dot_C0C1_C0C1 - dot_C0C1_C0O * dot_C0C1_D;
        double c = dot_C0O_C0O * dot_C0C1_C0C1 - dot_C0C1_C0O * dot_C0C1_C0O -
                   radius * radius * dot_C0C1_C0C1;
        double discriminant = half_b * half_b - a * c;
        bool hit_anything = false;
        double closest_so_far = ray_t.max;
        if (a > eps && discriminant >= -eps) {
            double sqrtd = std::sqrt(fmax(0.0, discriminant));
            double t = (-half_b - sqrtd) / a;
            vec3 C0P = r.origin() + t * r.direction() - base_center;
            double dot_C0P_C0C1 = dot(C0P, C0C1);
            // 0 <= y <= |C0C1|^2
            if (dot_C0P_C0C1 >= -height_eps && dot_C0P_C0C1 <= dot_C0C1_C0C1 + height_eps) {
                if (t >= ray_t.min && t <= closest_so_far) {
                    rec.t = t;
                    rec.p = r.at(t);
                    rec.material_id = material_id;
                    rec.set_face_normal(r, normalize(C0P - C0C1 * dot_C0P_C0C1 / dot_C0C1_C0C1));
                    hit_anything = true;
                    closest_so_far = t;
                }
            }
            t = (-half_b + sqrtd) / a;
            C0P = r.origin() + t * r.direction() - base_center;
            dot_C0P_C0C1 = dot(C0P, C0C1);
            if (dot_C0P_C0C1 >= -height_eps && dot_C0P_C0C1 <= dot_C0C1_C0C1 + height_eps) {
                if (t >= ray_t.min && t <= closest_so_far) {
                    rec.t = t;
                    rec.p = r.at(t);
                    rec.material_id = material_id;
                    rec.set_face_normal(r, normalize(C0P - C0C1 * dot_C0P_C0C1 / dot_C0C1_C0C1));
                    hit_anything = true;
                    closest_so_far = t;
                }
            }
        }

        // top cap
        vec3 top_normal = normalize(top_center - base_center);
        vec3 OC1 = top_center - r.origin();
        double dot_OC1_n1 = dot(OC1, top_normal);
        double dot_D_n1 = dot(r.direction(), top_normal);
        if (std::fabs(dot_D_n1) > eps) {
            double t = dot_OC1_n1 / dot_D_n1;
            vec3 C1P = r.origin() + t * r.direction() - top_center;
            if (t >= ray_t.min && t <= closest_so_far &&
                C1P.length_squared() <= radius * radius + eps) {
                rec.t = t;
                rec.p = r.at(t);
                rec.material_id = material_id;
                rec.set_face_normal(r, top_normal);
                hit_anything = true;
                closest_so_far = t;
            }
        }

        // base cap
        vec3 base_normal = -top_normal;
        vec3 OC0 = base_center - r.origin();
        double dot_OC0_n0 = dot(OC0, base_normal);
        double dot_D_n0 = dot(r.direction(), base_normal);
        if (std::fabs(dot_D_n0) > eps) {
            double t = dot_OC0_n0 / dot_D_n0;
            vec3 C0P = r.origin() + t * r.direction() - base_center;
            if (t >= ray_t.min && t <= closest_so_far &&
                C0P.length_squared() <= radius * radius + eps) {
                rec.t = t;
                rec.p = r.at(t);
                rec.material_id = material_id;
                rec.set_face_normal(r, base_normal);
                hit_anything = true;
                closest_so_far = t;
            }
        }

        return hit_anything;
    }

   private:
    point3 top_center, base_center;
    double height, radius;
    int material_id;
    aabb bbox;
};
