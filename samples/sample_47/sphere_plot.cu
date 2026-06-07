#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include "common.h"

struct point3 {
    double x, y, z;
};

__device__ inline point3 random_unit_sphere_point(curandState* state) {
    double r1 = curand_uniform_double(state);
    double r2 = curand_uniform_double(state);

    double phi = 2 * pi * r1;
    double sin_theta = 2 * std::sqrt(r2 * (1 - r2));
    double z = 1 - 2 * r2;
    
    return point3{
        .x = sin_theta * std::cos(phi),
        .y = sin_theta * std::sin(phi),
        .z = z
    };
}

__global__ void sphere_plot_kernel(point3* points, int n , unsigned long long seed){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    curandState state;
    curand_init(seed, i, 0, &state);
    points[i] = random_unit_sphere_point(&state);
}

int main() {
    const int n = 400;
    point3* d_points;
    cudaMalloc(&d_points, n * sizeof(point3));

    int block = 256;
    int grid = (n + block - 1) / block;
    sphere_plot_kernel<<<grid, block>>>(d_points, n, 19971122ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<point3> host_points(n);
    cudaMemcpy(host_points.data(), d_points, n * sizeof(point3), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; ++i) {
        std::cout << host_points[i].x << ' '
                  << host_points[i].y << ' '
                  << host_points[i].z << std::endl;
    }

    cudaFree(d_points);
    return 0;
}