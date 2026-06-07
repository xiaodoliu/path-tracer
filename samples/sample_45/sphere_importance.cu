#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <iomanip>

__device__ __host__ inline double pi_val() {return 3.14159265358979323846;}

__device__ inline double f(double z) {
    return z*z;
}

__device__ inline double pdf(double z) {
    return 3.0*z*z/(4.0*pi_val());
}

__global__ void integrate_kernel(double* partial, int total_threads,
                                 int samples_per_thread, unsigned long long seed){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= total_threads) return;

    curandState s;
    curand_init(seed, tid, 0, &s);
    
    double local_sum = 0.0;
    for(int k = 0; k < samples_per_thread; ++k){
        double u = curand_uniform_double(&s);
        double z = cbrt(2.0 * u - 1.0);
        local_sum += f(z) / pdf(z);
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