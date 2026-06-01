#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <iomanip>

__device__ __host__ inline double pi_val() {return 3.14159265358979323846;}

__global__ void generate_samples(double* xs, double* pxs, int N, unsigned long long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    curandState s;
    curand_init(seed, idx, 0, &s);

    double x = curand_uniform_double(&s) * 2 * pi_val();
    double sin_x = std::sin(x);
    double p_x = exp(-x/(2.0*pi_val())) * sin_x * sin_x;

    xs[idx] = x;
    pxs[idx] = p_x;
}

int main(){
    const unsigned int N = 15000;
    const double two_pi = 2.0 * pi_val();

    thrust::device_vector<double> d_x(N);
    thrust::device_vector<double> d_px(N);

    int block = 256;
    int grid = (N + block - 1) / block;

    generate_samples<<<grid, block>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_px.data()),
        N, /*seed=*/19971122ULL);
    cudaDeviceSynchronize();

    double sum = thrust::reduce(d_px.begin(), d_px.end(), 0.0);

    thrust::sort_by_key(d_x.begin(), d_x.end(), d_px.begin());

    thrust::device_vector<double> d_prefix(N);
    thrust::inclusive_scan(d_px.begin(), d_px.end(), d_prefix.begin());

    double halfway_sum = sum / 2.0;
    auto it = thrust::lower_bound(d_prefix.begin(), d_prefix.end(), halfway_sum);
    int halfway_idx = static_cast<int>(it - d_prefix.begin());
    double halfway_point = d_x[halfway_idx];

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Average = " << sum / N << std::endl;
    std::cout << "Area under curve = " << two_pi * sum / N << std::endl;
    std::cout << "Halfway point = " << halfway_point << std::endl;
    return 0;

}