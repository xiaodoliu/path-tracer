# pragma once

#include "vec3.h"
#include <vector>

typedef vec3 color;

D inline void write_color(unsigned char* image, int pixel_index, color pixel_color){
    image[pixel_index] = pixel_color.x() * 255.999;
    image[pixel_index + 1] = pixel_color.y() * 255.999;
    image[pixel_index + 2] = pixel_color.z() * 255.999;
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