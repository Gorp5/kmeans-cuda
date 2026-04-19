#include <cuda_runtime.h>
#include <stdio.h>

__global__ void findCentroid(float* data, int num_vecs, int n) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_id < num_vecs) {
        float* target_vector = data + data_id * n;

        // Find Centroid of cluster
    }
}

__global__ void assignCentroid(float* data, int num_vecs, int n) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_id < num_vecs) {
        float* target_vector = data + data_id * n;

        // Assign each point to a centroid
    }
}

int main() {
    int num_data = 1024;
    int data_size = 128;

    size_t total_size = num_data * data_size * sizeof(float);

    float* h_data = (float*)malloc(total_size);

    float* d_data;
    cudaMalloc(&d_data, total_size);

    int threads = 256;
    int blocks = (num_data + threads - 1) / threads;



    // Launch kernels
    cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);

    kernel1<<<blocks, threads>>>(d_data, num_data, data_size);
    kernel2<<<blocks, threads>>>(d_data, num_data, data_size);

    cudaMemcpy(h_data, d_data, total_size, cudaMemcpyDeviceToHost);



    cudaFree(d_data);
    free(h_data);

    return 0;
}