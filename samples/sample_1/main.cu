#include <cuda_runtime.h>
#include <iostream>

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
    unsigned char* host_image = new unsigned char[image_size];
    cudaMemcpy(host_image, image, image_size, cudaMemcpyDeviceToHost);
    cudaFree(image);
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for(size_t i = 0; i < image_size; ++i){
        std::cout << static_cast<int>(host_image[i]) << " ";
        if((i+1) % 3 == 0){
            std::cout << std::endl;
        }
    }
    delete[] host_image;
    return 0;
}