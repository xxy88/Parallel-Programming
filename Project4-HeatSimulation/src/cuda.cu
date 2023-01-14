#include "physics.h"
#include "shared.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


int block_size = 512;
int num_cuda_threads;


__global__ void initialize_cuda(float *data, int size) {
    int len = size * size;
    for (int i = 0; i < len; i++) {
        data[i] = wall_temp;
    }
}

__global__ void generate_fire_area_cuda(bool *fire_area, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size * size) {
        int i = idx / size;
        int j = idx % size;

        float fire1_r2 = fire_size * fire_size;
        int a = i - size / 2;
        int b = j - size / 2;
        int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
        if (r2 < fire1_r2)
            fire_area[i * size + j] = true;

        float fire2_r2 = (fire_size / 2) * (fire_size / 2);
        int c = i - 1 * size / 3;
        int d = j - 1 * size / 3;
        int r3 = c * c + d * d;
        if (r3 < fire2_r2)
            fire_area[i * size + j] = true;
    }
}


__global__ void update(float *data, float *new_data, int size, bool* fire_area) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx - size < 0 || idx + size >= size * size)
        return;
    if (fire_area[idx]) new_data[idx] = fire_temp;
    else {
        float up = data[idx - size];
        float down = data[idx + size];
        float left = data[idx - 1];
        float right = data[idx + 1];
        float new_val = (up + down + left + right) / 4;
        new_data[idx] = new_val;
    }
}


__global__ void maintain_wall(float *data, int size) {
    // maintain the temperature of the wall
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        data[idx] = wall_temp;
        data[idx * size] = wall_temp;
        data[idx * size + size - 1] = wall_temp;
        data[(size - 1) * size + idx] = wall_temp;
    }
}


void master() {
    float *device_data;
    float *device_new_data;
    bool *device_fire_area;
    bool *device_continue;

    data = new float[size * size];

    int len = size * size;
    int num_blocks = len % block_size ? len / block_size + 1 : len / block_size;
    num_cuda_threads = num_blocks * block_size;

    cudaMalloc(&device_data, len * sizeof(float));
    cudaMalloc(&device_new_data, len * sizeof(float));
    cudaMalloc(&device_fire_area, len * sizeof(bool));
    cudaMalloc(&device_continue, sizeof(bool));

    initialize_cuda<<<1, 1>>>(device_data, size);
    generate_fire_area_cuda<<<num_blocks, block_size>>>(device_fire_area, size);

    while (count <= max_iteration) {
        t1 = std::chrono::high_resolution_clock::now();

        update<<<num_blocks, block_size>>>(device_data, device_new_data, size, device_fire_area);
        maintain_wall<<<num_blocks, block_size>>>(device_new_data, size);
        swap(device_data, device_new_data);
//        cudaMemcpy(device_data, device_new_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
//        cudaDeviceSynchronize();

        t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        if (DEBUG) printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        total_time += this_time;

#ifdef GUI
        data2pixels(data, pixels);
        plot(pixels);
#endif

        count++;
    }

    printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count - 1, total_time,
           total_time / (count - 1));

    cudaFree(device_data);
    cudaFree(device_new_data);
    cudaFree(device_fire_area);
    cudaFree(device_continue);

    delete[] data;
}


int main(int argc, char *argv[]) {
    size = atoi(argv[1]);

#ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_size, window_size);
    glutCreateWindow("Heat Distribution Simulation CUDA Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
#endif

    master();

    printf("Student ID: 119020059\n");
    printf("Name: Xinyu Xie\n");
    printf("Assignment 4: Heat Distribution CUDA Implementation\n");
    printf("Problem Size: %d\n", size);
    printf("Number of Cores: %d\n", num_cuda_threads);

    return 0;
}
