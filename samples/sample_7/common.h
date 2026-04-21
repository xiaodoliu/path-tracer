# pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#define H __host__
#define D __device__
#else
#define HD 
#define H 
#define D 
#endif

inline constexpr double infinity = std::numeric_limits<double>::infinity();
inline constexpr double pi = 3.1415926535897932385;

HD inline double degrees_to_radians(double degrees){
    return degrees * pi / 180.0;
}

