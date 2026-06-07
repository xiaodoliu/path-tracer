#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <iomanip>

// f(x)/pdf(x) with f(x) = x^2 and pdf(x) = 0.5
__device__ inline double sample_value(double x){
    return x*x / (0.5);
}

__global__ void integrate_kernel(double* partial, int total_threads, 
                                 int samples_per_thread, unsigned long long seed){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= total_threads) return;

    curandState s;
    curand_init(seed, tid, 0, &s);

    double local_sum = 0.0;
    for(int k = 0; k < samples_per_thread; ++k){
        double d = curand_uniform_double(&s);
        double x = 2.0 * d;
        local_sum += sample_value(x);
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
    std::cout << "I = " << sum / (double)(total_threads * samples_per_thread) << std::endl;
    return 0;
}
