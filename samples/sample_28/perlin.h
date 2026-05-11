#pragma once

#include "common.h"

class perlin {
public:
    perlin(){
        for(int i = 0; i < point_count; ++i){
            randfloat[i] = random_double();
        }
        perlin_generate_perm(perm_x);
        perlin_generate_perm(perm_y);
        perlin_generate_perm(perm_z);
    }
    
    D double noise(const point3& p) const{
        auto i = int(4*p.x()) & (point_count-1);
        auto j = int(4*p.y()) & (point_count-1);
        auto k = int(4*p.z()) & (point_count-1);
        return randfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
    }

private:
    static const int point_count = 256;
    double randfloat[point_count];
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
};