# pragma once

#include "hittable.h"

class quad: public hittable{
public:
    quad() = default;
    quad(const point3& Q, const vec3& u, const vec3& v,
      int material_id): Q(Q), u(u), v(v), material_id(material_id){
        auto n = cross(u,v);
        w = n/dot(n,n);
        normal = normalize(n);
        D_N_P0 = dot(normal, Q);
        set_bounding_box();
    }

    virtual void set_bounding_box() {
        auto bbox_diagonal1 = aabb(Q, Q+u+v);
        auto bbox_diagonal2 = aabb(Q+u, Q+v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    HD aabb bounding_box() const override {return bbox;}

    D bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        auto denom = dot(normal, r.direction());
        if(std::fabs(denom) < 1e-8) return false;
        auto t = (D_N_P0 - dot(normal, r.origin())) / denom;
        if(!ray_t.contains(t)) return false;
        auto intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));
        if(!is_interior(alpha, beta, rec)) return false;
        rec.t = t;
        rec.p = intersection;
        rec.material_id = material_id;
        rec.set_face_normal(r, normal);
        return true;
    }

    D bool is_interior(double alpha, double beta, hit_record& rec) const{
        interval unit_interval(0, 1);
        if(!unit_interval.contains(alpha) || !unit_interval.contains(beta)) return false;
        rec.u = alpha;
        rec.v = beta;
        return true;
    }
    
private:
    point3 Q;
    vec3 u, v;
    vec3 w; // n/n^2
    int material_id;
    aabb bbox;
    vec3 normal;
    double D_N_P0;
};