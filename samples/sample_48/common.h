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

#include <random>
#include <limits>
#include <curand_kernel.h>

#define CUDA_CHECK(call) \
    do{ \
        cudaError_t err = (call); \
        if(err != cudaSuccess){ \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        } \
    }while(0)

inline constexpr double infinity = std::numeric_limits<double>::infinity();
inline constexpr double pi = 3.1415926535897932385;

HD inline double degrees_to_radians(double degrees){
    return degrees * pi / 180.0;
}

// returns a random real in [0,1)
H inline double random_double(){
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

H inline double random_double(double min, double max){
    return min + (max-min)*random_double();
}

// returns a random integer in [min, max]
H inline int random_int(int min, int max){
    return static_cast<int>(random_double(min, max+1));
}

D inline double random_double(curandState* state){
    return curand_uniform(state);
}

D inline double random_double(double min, double max, curandState* state){
    return min + (max-min)*random_double(state);
}
