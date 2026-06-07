#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <iomanip>

#include "common.h"

__device__ inline double f(double r2){
    auto z = 1 - r2;
    double cos_theta = z;
    return cos_theta * cos_theta * cos_theta;
}

__device__ inline double pdf(){
    return 1 / (2 * pi);
}

__global__ void integrate_kernel(double* partial, int total_threads, 
                                 int samples_per_thread, unsigned long long seed){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= total_threads) return;

    curandState s;
    curand_init(seed, tid, 0, &s);
    double local_sum = 0.0;
    for(int k = 0; k < samples_per_thread; ++k){
        double r2 = curand_uniform_double(&s);
        local_sum += f(r2) / pdf();
    }
    partial[tid] = local_sum;
}

int main(){
    const long long N = 1000000;
    const int samples_per_thread = 100;
    const int total_threads = N / samples_per_thread;

    thrust::device_vector<double> d_partial(total_threads);

    int block = 256;
    int grid = (total_threads + block - 1) / block;
    integrate_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_partial.data()),
        total_threads, samples_per_thread, /*seed=*/19971122ULL);
    cudaDeviceSynchronize();

    double sum = thrust::reduce(d_partial.begin(), d_partial.end(), 0.0);
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "PI / 2 = " << pi/2.0 << std::endl;
    std::cout << "Estimate = " << sum / (double)(total_threads * samples_per_thread) << std::endl;
    return 0;
}
