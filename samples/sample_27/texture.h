#pragma once

#include "rtw_stb_image.h"

enum class texture_type {
    SOLID_COLOR,
    CHECKER,
    IMAGE,
};

struct texture_data{
    texture_type type;
    color color1;
    color color2;
    double inv_scale = 1.0;
    unsigned char* image_data = nullptr;
    int image_width = 0;
    int image_height = 0;
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
        case texture_type::IMAGE:{
            auto& image_data = tex.image_data;
            auto& image_width = tex.image_width;
            auto& image_height = tex.image_height;
            if(image_height <= 0) return color(0,1,1);
            u = interval(0, 1).clamp(u);
            v = 1.0 - interval(0, 1).clamp(v);
            auto i = static_cast<int>(u * image_width);
            auto j = static_cast<int>(v * image_height);
            auto pixel = image_data + j * image_width * 3 + i * 3;
            auto color_scale = 1/255.0;
            return color(pixel[0] * color_scale, pixel[1] * color_scale, pixel[2] * color_scale);
        }
        default:
            return color(0.0, 0.0, 0.0);
    }}
