#pragma once

#include "hittable.h"

#include <cassert>

// A vertical elliptical paraboloid with its largest ellipse at base_y and a
// single peak at base_y + height:
//
//   (x-cx)^2 / radius_x^2 + (z-cz)^2 / radius_z^2
//       + (y-base_y) / height - 1 = 0
class paraboloid {
   public:
    H paraboloid() = default;

    H paraboloid(const point3& base_center, double radius_x, double radius_z, double height,
                 int material_id)
        : base_center(base_center),
          radius_x(radius_x),
          radius_z(radius_z),
          height(height),
          inv_radius_x_squared(1.0 / (radius_x * radius_x)),
          inv_radius_z_squared(1.0 / (radius_z * radius_z)),
          inv_height(1.0 / height),
          material_id(material_id),
          bbox(point3(base_center.x() - radius_x, base_center.y(), base_center.z() - radius_z),
               point3(base_center.x() + radius_x, base_center.y() + height,
                      base_center.z() + radius_z)) {
        assert(radius_x > 0.0);
        assert(radius_z > 0.0);
        assert(height > 0.0);
    }

    HD aabb bounding_box() const { return bbox; }

    D bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        constexpr double eps = 1e-12;
        vec3 origin = r.origin() - base_center;
        const vec3& direction = r.direction();

        double a = direction.x() * direction.x() * inv_radius_x_squared +
                   direction.z() * direction.z() * inv_radius_z_squared;
        double b = 2.0 * origin.x() * direction.x() * inv_radius_x_squared +
                   2.0 * origin.z() * direction.z() * inv_radius_z_squared +
                   direction.y() * inv_height;
        double c = origin.x() * origin.x() * inv_radius_x_squared +
                   origin.z() * origin.z() * inv_radius_z_squared +
                   origin.y() * inv_height - 1.0;

        bool hit_anything = false;
        double closest = ray_t.max;
        double roots[2];
        int root_count = 0;

        if (std::fabs(a) > eps) {
            double discriminant = b * b - 4.0 * a * c;
            if (discriminant >= 0.0) {
                double square_root = std::sqrt(discriminant);
                roots[0] = (-b - square_root) / (2.0 * a);
                roots[1] = (-b + square_root) / (2.0 * a);
                root_count = 2;
            }
        } else if (std::fabs(b) > eps) {
            // A ray parallel to the paraboloid axis produces a linear equation.
            roots[0] = -c / b;
            root_count = 1;
        }

        for (int root_index = 0; root_index < root_count; ++root_index) {
            double t = roots[root_index];
            if (t < ray_t.min || t > closest) continue;

            point3 p = r.at(t);
            double local_y = p.y() - base_center.y();
            if (local_y < -eps || local_y > height + eps) continue;

            vec3 local = p - base_center;
            vec3 outward_normal(
                2.0 * local.x() * inv_radius_x_squared,
                inv_height,
                2.0 * local.z() * inv_radius_z_squared);
            rec.t = t;
            rec.p = p;
            rec.material_id = material_id;
            rec.u = std::atan2(local.z() / radius_z, local.x() / radius_x) / (2.0 * pi) + 0.5;
            rec.v = std::fmax(0.0, std::fmin(1.0, local_y * inv_height));
            rec.set_face_normal(r, outward_normal);
            hit_anything = true;
            closest = t;
        }

        // Close the open bottom with an elliptical cap. Mountain bases are
        // placed below the ground, but the cap keeps intersections watertight.
        if (std::fabs(direction.y()) > eps) {
            double t = -origin.y() / direction.y();
            if (t >= ray_t.min && t <= closest) {
                point3 p = r.at(t);
                double local_x = p.x() - base_center.x();
                double local_z = p.z() - base_center.z();
                double ellipse = local_x * local_x * inv_radius_x_squared +
                                 local_z * local_z * inv_radius_z_squared;
                if (ellipse <= 1.0 + eps) {
                    rec.t = t;
                    rec.p = p;
                    rec.material_id = material_id;
                    rec.u = local_x / (2.0 * radius_x) + 0.5;
                    rec.v = local_z / (2.0 * radius_z) + 0.5;
                    rec.set_face_normal(r, vec3(0.0, -1.0, 0.0));
                    hit_anything = true;
                }
            }
        }

        return hit_anything;
    }

   private:
    point3 base_center;
    double radius_x;
    double radius_z;
    double height;
    double inv_radius_x_squared;
    double inv_radius_z_squared;
    double inv_height;
    int material_id;
    aabb bbox;
};
