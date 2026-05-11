#pragma once

#include "common.h"
#include "vec3.h"

class perlin {
public:
    perlin(){
        for(int i = 0; i < point_count; ++i){
            randvec[i] = normalize(vec3::random(-1.0, 1.0));
        }
        perlin_generate_perm(perm_x);
        perlin_generate_perm(perm_y);
        perlin_generate_perm(perm_z);
    }
    
    D double noise(const point3& p) const{
        auto u = p.x() - std::floor(p.x());
        auto v = p.y() - std::floor(p.y());
        auto w = p.z() - std::floor(p.z());
        u = u * u * (3 - 2 * u);
        v = v * v * (3 - 2 * v);
        w = w * w * (3 - 2 * w);
        auto i = int(std::floor(p.x()));
        auto j = int(std::floor(p.y()));
        auto k = int(std::floor(p.z()));
        vec3 c[2][2][2];
        int point_count_minus_one = point_count - 1;
        for(int di = 0; di < 2; ++di){
            for(int dj = 0; dj < 2; ++dj){
                for(int dk = 0; dk < 2; ++dk){
                    c[di][dj][dk] = 
                        randvec[perm_x[(i+di) & point_count_minus_one]
                            ^ perm_y[(j+dj) & point_count_minus_one]
                            ^ perm_z[(k+dk) & point_count_minus_one]];
                }
            }
        }
        return perlin_interp(c, u, v, w);
    }

    D double turb(const point3& p, int depth) const{
        auto accum = 0.0;
        auto temp_p = p;
        auto weight = 1.0;
        for(int i = 0; i < depth; ++i){
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2;
        }
        return std::fabs(accum);
    }

private:
    static const int point_count = 256;
    vec3 randvec[point_count];
    int perm_x[point_count];
    int perm_y[point_count];
    int perm_z[point_count];
    
    static void perlin_generate_perm(int* p){
        for(int i = 0; i < point_count; ++i){
            p[i] = i;
        }
        permute(p, point_count);
    }

    static void permute(int* p, int n){
        for(int i = n-1; i > 0; --i){
            int target = random_int(0, i);
            std::swap(p[i], p[target]);
        }
    }

    // D static double trilinear_interp(double c[2][2][2], double u, double v, double w) {
    //     double accum = 0.0;
    //     for(int i = 0; i < 2; ++i){
    //         for(int j = 0; j < 2; ++j){
    //             for(int k = 0; k < 2; ++k){
    //                 accum += (i*u + (1-i)*(1-u)) * (j*v + (1-j)*(1-v)) * (k*w + (1-k)*(1-w)) * c[i][j][k];
    //             }
    //         }
    //     }
    //     return accum;
    // }

    D static double perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
        double accum = 0.0;
        for(int i = 0; i < 2; ++i){
            for(int j = 0; j < 2; ++j){
                for(int k = 0; k < 2; ++k){
                    vec3 weight_v(u-i, v-j, w-k);
                    accum += (i*u + (1-i)*(1-u)) 
                        * (j*v + (1-j)*(1-v)) 
                        * (k*w + (1-k)*(1-w)) 
                        * dot(c[i][j][k], weight_v);
                }
            }
        }
        return accum;
    }
};