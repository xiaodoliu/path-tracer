#pragma once

#include "quad.h"

struct box {
    quad sides[6];
    aabb bbox;

    HD box() = default;

    HD box(const point3& a, const point3& b, int material_id) {
        auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

        auto dx = vec3(max.x() - min.x(), 0, 0);
        auto dy = vec3(0, max.y() - min.y(), 0);
        auto dz = vec3(0, 0, max.z() - min.z());

        sides[0] = quad(min, dy, dx, material_id);    // back
        sides[1] = quad(min, dx, dz, material_id);    // bottem
        sides[2] = quad(min, dz, dy, material_id);    // left
        sides[3] = quad(max, -dx, -dy, material_id);  // front
        sides[4] = quad(max, -dz, -dx, material_id);  // top
        sides[5] = quad(max, -dy, -dz, material_id);  // right

        bbox = aabb(min, max);
    }

    HD box(const point3& a, const point3& b, int material_id, double angle, const vec3& offset) {
        auto min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        auto max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));
        auto dx = vec3(max.x() - min.x(), 0, 0);
        auto dy = vec3(0, max.y() - min.y(), 0);
        auto dz = vec3(0, 0, max.z() - min.z());
        sides[0] = quad(rotate_y(min, angle) + offset, rotate_y(dy, angle), rotate_y(dx, angle),
                        material_id);
        sides[1] = quad(rotate_y(min, angle) + offset, rotate_y(dx, angle), rotate_y(dz, angle),
                        material_id);
        sides[2] = quad(rotate_y(min, angle) + offset, rotate_y(dz, angle), rotate_y(dy, angle),
                        material_id);
        sides[3] = quad(rotate_y(max, angle) + offset, rotate_y(-dx, angle), rotate_y(-dy, angle),
                        material_id);
        sides[4] = quad(rotate_y(max, angle) + offset, rotate_y(-dz, angle), rotate_y(-dx, angle),
                        material_id);
        sides[5] = quad(rotate_y(max, angle) + offset, rotate_y(-dy, angle), rotate_y(-dz, angle),
                        material_id);

        bbox = sides[0].bounding_box();
        for (int i = 1; i < 6; ++i) {
            bbox = aabb(bbox, sides[i].bounding_box());
        }
    }

    D bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = ray_t.max;
        for (int i = 0; i < 6; ++i) {
            if (sides[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    HD aabb bounding_box() const { return bbox; }
};