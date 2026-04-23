#include <cuda_runtime.h>
#include <stdio.h>

// convert to using the nice cuda types later
// Dot product optimization can be done using: ||x−c||^2 = ||x||^2 + ||c||^2 − 2(x * c)
__global__ void assignCentroid(ui* assignments, float* data, int num_vecs, float* centroids, ui k, int dim) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;

    float min_distance = FLT_MAX;
    int min_k = -1;

    float* target_vector = data + data_id * dim;

    if (data_id >= num_vecs) return;
    for (int current_k = 0; current_k < k; current_k++) {

        // Distance Operation
        float running_sum = 0;
        for (int current_dim = 0; current_dim < dim; current_dim++) {
            float distance = target_vector[current_dim] - centroids[current_k * dim + current_dim];
            running_sum += distance * distance;
        }

        if (running_sum < min_distance) {
            min_distance = running_sum;
            min_k = current_k;
        }
    }

    assignments[data_id] = data + data_id * k;
}

__global__ void find_average_centroid(float* data, int num_vecs, int n) {
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


    assignCentroid<<<blocks, threads>>>(d_data, num_data, data_size);
    findCentroid<<<blocks, threads>>>(d_data, num_data, data_size);


    cudaMemcpy(h_data, d_data, total_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    free(h_data);

    return 0;
}