#include "camera.h"
#include "color.h"
#include "ray.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

D double hit_sphere(const point3 &center, double radius, const ray &r) {
  vec3 oc = center - r.origin();
  auto a = r.direction().length_squared();
  auto h = dot(oc, r.direction());
  auto c = oc.length_squared() - radius * radius;
  auto discriminant = h * h - a * c;
  if (discriminant < 0.0) {
    return -1.0;
  }
  return h - std::sqrt(discriminant);
}

D color ray_color(const ray &r) {
  auto t = hit_sphere(point3(0, 0, -1), 0.5, r);
  if (t >= 0.0) {
    vec3 N = normalize(r.at(t) - point3(0, 0, -1));
    return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
  }
  vec3 unit_direction = normalize(r.direction());
  t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render_kernel(unsigned char *image, int width, int height,
                              camera cam) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height)
    return;
  auto pixel_center = cam.pixel_sample_start + i * cam.viewport_pixel_u_delta +
                      j * cam.viewport_pixel_v_delta;
  auto ray_direction = pixel_center - cam.origin;
  ray r(cam.origin, ray_direction);
  color pixel_color = ray_color(r);
  int pixel_index = (j * width + i) * 3;
  write_color(image, pixel_index, pixel_color);
}

int main() {
  double aspect_ratio = 16.0 / 9.0;
  int image_width = 800;
  int image_height = static_cast<int>(image_width / aspect_ratio);
  image_height = (image_height < 1) ? 1 : image_height;
  double focal_length = 1.0;
  double viewport_height = 2.0;
  double viewport_width =
      viewport_height * static_cast<double>(image_width) / image_height;
  vec3 viewport_u(viewport_width, 0, 0);
  vec3 viewport_v(0, -viewport_height, 0);
  vec3 viewport_pixel_u_delta = viewport_u / image_width;
  vec3 viewport_pixel_v_delta = viewport_v / image_height;
  camera cam;
  cam.origin = point3(0, 0, 0);
  cam.focal_length = focal_length;
  cam.viewport_pixel_u_delta = viewport_pixel_u_delta;
  cam.viewport_pixel_v_delta = viewport_pixel_v_delta;
  auto viewport_upper_left =
      cam.origin - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
  cam.pixel_sample_start = viewport_upper_left + viewport_pixel_u_delta / 2 +
                           viewport_pixel_v_delta / 2;

  unsigned char *image;
  size_t image_size = image_width * image_height * 3;
  cudaMalloc(&image, image_size);
  dim3 block_size = dim3(16, 16);
  dim3 grid_size = dim3((image_width + block_size.x - 1) / block_size.x,
                        (image_height + block_size.y - 1) / block_size.y);
  render_kernel<<<grid_size, block_size>>>(image, image_width, image_height,
                                           cam);
  cudaDeviceSynchronize();
  std::vector<unsigned char> host_image(image_size);
  cudaMemcpy(host_image.data(), image, image_size, cudaMemcpyDeviceToHost);
  cudaFree(image);
  write_image(std::cout, host_image, image_width, image_height);
  return 0;
}
