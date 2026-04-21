#include "color.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Kernel to render an image, rgb ranging from (0, 0, 0) to (255, 255, 0).
__global__ void render_kernel(unsigned char* image, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) return;
    int pixel_index = (y * width + x) * 3;
    image[pixel_index] = x * 255.999 / (width - 1);
    image[pixel_index + 1] = y * 255.999 / (height - 1);
    image[pixel_index + 2] = 0;
}


int main(){
    int width = 256, height = 256;
    unsigned char* image;
    size_t image_size = width * height * 3;
    cudaMalloc(&image, image_size);
    dim3 block_size = dim3(16, 16);
    dim3 grid_size = dim3(width / block_size.x, height / block_size.y);
    render_kernel<<<grid_size, block_size>>>(image, width, height);
    cudaDeviceSynchronize();
    std::vector<unsigned char> host_image(image_size);
    cudaMemcpy(host_image.data(), image, image_size, cudaMemcpyDeviceToHost);
    cudaFree(image);
    write_image(std::cout, host_image, width, height);
    return 0;
}