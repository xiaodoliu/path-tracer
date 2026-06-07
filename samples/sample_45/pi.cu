#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>

__global__ void init_rand_state(curandState* states, int sqrt_N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= sqrt_N || j >= sqrt_N) return;
    int idx = j * sqrt_N + i;
    curand_init(/*seed=*/19971122, idx, 0, &states[idx]);
}

__global__ void estimate_pi_kernel(
    int sqrt_N, 
    curandState* states, 
    unsigned long long* inside_uniform, 
    unsigned long long* inside_stratified){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= sqrt_N || j >= sqrt_N) return;
    int idx = j * sqrt_N + i;
    curandState s = states[idx];
    
    // uniform sampling
    double rx = curand_uniform_double(&s) * 2.0 - 1.0;
    double ry = curand_uniform_double(&s) * 2.0 - 1.0;
    if(rx * rx + ry * ry < 1.0){
        atomicAdd(inside_uniform, 1ULL);
    }

    // stratified sampling
    rx = 2.0 * (i + curand_uniform_double(&s)) / sqrt_N - 1.0;
    ry = 2.0 * (j + curand_uniform_double(&s)) / sqrt_N - 1.0;
    if(rx * rx + ry * ry < 1.0){
        atomicAdd(inside_stratified, 1ULL);
    }
    states[idx] = s;
}

int main(){
    std::cout << std::fixed << std::setprecision(12);

    int sqrt_N = 3000;
    int total = sqrt_N * sqrt_N;
    curandState* d_states;
    cudaMalloc(&d_states, total * sizeof(curandState));

    unsigned long long* d_inside_uniform;
    unsigned long long* d_inside_stratified;

    cudaMalloc(&d_inside_uniform, sizeof(unsigned long long));
    cudaMalloc(&d_inside_stratified, sizeof(unsigned long long));
    cudaMemset(d_inside_uniform, 0, sizeof(unsigned long long));
    cudaMemset(d_inside_stratified, 0, sizeof(unsigned long long));

    dim3 block(16, 16);
    dim3 grid(
        (sqrt_N + block.x - 1) / block.x,
        (sqrt_N + block.y - 1) / block.y
    );

    init_rand_state<<<grid, block>>>(d_states, sqrt_N);
    estimate_pi_kernel<<<grid, block>>>(sqrt_N, d_states, d_inside_uniform, d_inside_stratified);
    cudaDeviceSynchronize();

    unsigned long long h_inside_uniform = 0, h_inside_stratified = 0;
    cudaMemcpy(&h_inside_uniform, d_inside_uniform, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_inside_stratified, d_inside_stratified, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "regular estimate of pi = " << 4.0 * h_inside_uniform / total << std::endl;
    std::cout << "stratified estimate of pi = " << 4.0 * h_inside_stratified / total << std::endl;

    cudaFree(d_states);
    cudaFree(d_inside_uniform);
    cudaFree(d_inside_stratified);

    return 0;
}