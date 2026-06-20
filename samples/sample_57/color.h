# pragma once

#include "vec3.h"
#include "interval.h"
#include <vector>

typedef vec3 color;

HD inline double linear_to_gamma(double linear_component){
    if(linear_component < 0.0) return 0.0;
    return std::sqrt(linear_component);
}

D inline void write_color(unsigned char* image, int pixel_index, color pixel_color){
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Replace NaN components with 0.
    if(std::isnan(r)) r = 0.0;
    if(std::isnan(g)) g = 0.0;
    if(std::isnan(b)) b = 0.0;

    const interval intensity(0.0, 1.0);    
    image[pixel_index] = static_cast<int>(255.999 * intensity.clamp(linear_to_gamma(r)));
    image[pixel_index + 1] = static_cast<int>(255.999 * intensity.clamp(linear_to_gamma(g)));
    image[pixel_index + 2] = static_cast<int>(255.999 * intensity.clamp(linear_to_gamma(b)));
}

// Print out image data as ppm format.
inline void write_image(std::ostream& out, std::vector<unsigned char>& image, int width, int height){
    out << "P3\n" << width << " " << height << "\n255\n";
    for(size_t i = 0; i < image.size(); ++i){
        out << static_cast<int>(image[i]) << " ";
        if((i+1)%3 == 0){
            out << "\n";
        }
    }
    out << std::endl;
}