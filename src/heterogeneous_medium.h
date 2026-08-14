#pragma once

#include "hittable.h"

// A cloud is represented by one heterogeneous participating medium. The lobes
// define its large silhouette while procedural noise cuts holes into their
// union and erodes the edge, avoiding the appearance of overlapping fog balls.
struct heterogeneous_medium {
    static constexpr int max_lobes = 24;

    struct lobe {
        point3 center;
        double radius;
        double density_scale;
        bool lighter;
    };

    aabb bbox;
    point3 noise_origin;
    double base_density;
    double max_density;
    double cloud_base_y;
    int dark_material_id;
    int light_material_id;
    int lobe_count;
    lobe lobes[max_lobes];

    HD heterogeneous_medium()
        : base_density(0.0),
          max_density(0.0),
          cloud_base_y(0.0),
          dark_material_id(-1),
          light_material_id(-1),
          lobe_count(0) {}

    HD heterogeneous_medium(const point3& bounds_min, const point3& bounds_max,
                            const point3& origin, double density, double base_y,
                            int dark_material, int light_material)
        : bbox(bounds_min, bounds_max),
          noise_origin(origin),
          base_density(density),
          max_density(density * 1.4),
          cloud_base_y(base_y),
          dark_material_id(dark_material),
          light_material_id(light_material),
          lobe_count(0) {}

    H void add_lobe(const point3& center, double radius, double density_scale, bool lighter) {
        if (lobe_count < max_lobes) {
            lobes[lobe_count++] = {center, radius, density_scale, lighter};
        }
    }

    HD static double clamp01(double value) {
        return fmin(1.0, fmax(0.0, value));
    }

    HD static double smoothstep(double edge0, double edge1, double value) {
        double t = clamp01((value - edge0) / (edge1 - edge0));
        return t * t * (3.0 - 2.0 * t);
    }

    HD static unsigned int hash(unsigned int value) {
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        value *= 0x846ca68bU;
        return value ^ (value >> 16);
    }

    HD static double lattice_noise(int x, int y, int z) {
        unsigned int h = hash(static_cast<unsigned int>(x) * 0x8da6b343U ^
                              static_cast<unsigned int>(y) * 0xd8163841U ^
                              static_cast<unsigned int>(z) * 0xcb1ab31fU);
        return (h & 0x00ffffffU) / 16777215.0;
    }

    HD static double value_noise(const point3& p) {
        int x0 = static_cast<int>(floor(p.x()));
        int y0 = static_cast<int>(floor(p.y()));
        int z0 = static_cast<int>(floor(p.z()));
        double tx = p.x() - x0;
        double ty = p.y() - y0;
        double tz = p.z() - z0;
        tx = tx * tx * (3.0 - 2.0 * tx);
        ty = ty * ty * (3.0 - 2.0 * ty);
        tz = tz * tz * (3.0 - 2.0 * tz);

        double c000 = lattice_noise(x0,     y0,     z0);
        double c100 = lattice_noise(x0 + 1, y0,     z0);
        double c010 = lattice_noise(x0,     y0 + 1, z0);
        double c110 = lattice_noise(x0 + 1, y0 + 1, z0);
        double c001 = lattice_noise(x0,     y0,     z0 + 1);
        double c101 = lattice_noise(x0 + 1, y0,     z0 + 1);
        double c011 = lattice_noise(x0,     y0 + 1, z0 + 1);
        double c111 = lattice_noise(x0 + 1, y0 + 1, z0 + 1);

        double x00 = c000 + tx * (c100 - c000);
        double x10 = c010 + tx * (c110 - c010);
        double x01 = c001 + tx * (c101 - c001);
        double x11 = c011 + tx * (c111 - c011);
        double y0_value = x00 + ty * (x10 - x00);
        double y1_value = x01 + ty * (x11 - x01);
        return y0_value + tz * (y1_value - y0_value);
    }

    HD static double cloud_noise(const point3& p) {
        // Low frequencies form broad cavities; the last octave roughens only
        // the silhouette enough to remain stable at animation resolution.
        return 0.55 * value_noise(0.18 * p) +
               0.30 * value_noise(0.43 * p + vec3(11.7, 4.1, 8.3)) +
               0.15 * value_noise(1.05 * p + vec3(3.2, 19.1, 5.7));
    }

    HD double density_at(const point3& p) const {
        double coverage = 0.0;
        double lobe_density_scale = 0.0;
        for (int i = 0; i < lobe_count; ++i) {
            vec3 offset = (p - lobes[i].center) / lobes[i].radius;
            double candidate = 1.0 - offset.length_squared();
            if (candidate > coverage) {
                coverage = candidate;
                lobe_density_scale = lobes[i].density_scale;
            }
        }
        if (coverage <= 0.0) {
            return 0.0;
        }

        point3 local_p = p - noise_origin;
        double noise = cloud_noise(local_p);

        // Noise erodes the nominal lobe boundary. The smooth ramp produces
        // wisps instead of a hard procedural cutout.
        double eroded_shape = coverage + 0.42 * (noise - 0.5);
        double edge = smoothstep(0.015, 0.32, eroded_shape);

        // Cumulus/storm clouds tend to have a comparatively level base. Keep
        // that transition sharper than the noisy, billowing upper surface.
        double base_profile = smoothstep(cloud_base_y, cloud_base_y + 0.55, p.y());
        double interior_variation = 0.58 + 0.42 * noise;
        return base_density * lobe_density_scale * edge * base_profile * interior_variation;
    }

    D int material_at(const point3& p) const {
        double best_coverage = -infinity;
        int best_lobe = 0;
        for (int i = 0; i < lobe_count; ++i) {
            vec3 offset = (p - lobes[i].center) / lobes[i].radius;
            double coverage = 1.0 - offset.length_squared();
            if (coverage > best_coverage) {
                best_coverage = coverage;
                best_lobe = i;
            }
        }
        return lobes[best_lobe].lighter ? light_material_id : dark_material_id;
    }

    D bool ray_bounds(const ray& r, interval ray_t, double& entry, double& exit) const {
        entry = ray_t.min;
        exit = ray_t.max;
        for (int axis = 0; axis < 3; ++axis) {
            const interval& bounds = bbox.axis_interval(axis);
            double direction = r.direction()[axis];
            if (fabs(direction) < 1e-12) {
                if (r.origin()[axis] < bounds.min || r.origin()[axis] > bounds.max) {
                    return false;
                }
                continue;
            }
            double t0 = (bounds.min - r.origin()[axis]) / direction;
            double t1 = (bounds.max - r.origin()[axis]) / direction;
            if (t0 > t1) {
                double swap = t0;
                t0 = t1;
                t1 = swap;
            }
            entry = fmax(entry, t0);
            exit = fmin(exit, t1);
            if (exit <= entry) {
                return false;
            }
        }
        return true;
    }

    // Woodcock/delta tracking samples a varying density without stepping at a
    // fixed interval. max_density is a majorant for every procedural sample.
    D bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const {
        double entry;
        double exit;
        if (max_density <= 0.0 || !ray_bounds(r, ray_t, entry, exit)) {
            return false;
        }

        double ray_length = r.direction().length();
        double t = fmax(entry, 0.0);
        for (int iteration = 0; iteration < 256; ++iteration) {
            double random_value = fmax(random_double(state), 1e-12);
            t += -log(random_value) / (max_density * ray_length);
            if (t >= exit) {
                return false;
            }

            point3 p = r.at(t);
            double local_density = density_at(p);
            if (random_double(state) * max_density < local_density) {
                rec.t = t;
                rec.p = p;
                // Volumes have no surface normal. Store the incident axis here
                // so the phase-function sampler can orient forward scattering.
                rec.normal = normalize(r.direction());
                rec.front_face = true;
                rec.u = 0.0;
                rec.v = 0.0;
                rec.material_id = material_at(p);
                return true;
            }
        }
        return false;
    }

    HD aabb bounding_box() const { return bbox; }
};
