# pragma once

#include "vec3.h"
#include <vector>

typedef vec3 color;

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