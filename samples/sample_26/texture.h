#pragma once

enum class texture_type {
    SOLID_COLOR,
    CHECKER,
};

struct texture_data{
    texture_type type;
    color color1;
    color color2;
    double inv_scale = 1.0;
};

D inline color texture_value(double u, double v, const point3& p, const texture_data& tex){
    switch(tex.type){
        case texture_type::SOLID_COLOR:
            return tex.color1;
        case texture_type::CHECKER:{
            auto xi = static_cast<int>(std::floor(tex.inv_scale * p.x()));
            auto yi = static_cast<int>(std::floor(tex.inv_scale * p.y()));
            auto zi = static_cast<int>(std::floor(tex.inv_scale * p.z()));
            bool is_even = (xi + yi + zi) % 2 == 0;
            return is_even ? tex.color1 : tex.color2;
        }
        default:
            return color(0.0, 0.0, 0.0);
    }
}