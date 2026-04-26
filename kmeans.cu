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

__global__ void dotProductData(float* data, int num_vecs, float* centroids, ui k, int dim, float* results) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;
    int target_k = blockIdx.x * blockDim.x + threadIdx.y;

    if (data_id >= num_vecs || target_k >= k) return;

    const float* x = &data[data_id * dim];
    const float* c = &centroids[target_k * dim];

    float running_sum = 0;

    for (int index = 0; index < dim; index++) {
        running_sum += x[index] * c[index];
    }

    results[data_id * k + target_k] = running_sum;
}

__global__ void dotProductCentroid(float* centroids, ui k, int dim, float* results) {
    int target_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (target_k >= k) return;

    const float* c = &centroids[target_k * dim];

    float running_sum = 0;

    for (int index = 0; index < dim; index++) {
        running_sum += c[index] * c[index];
    }

    results[target_k] = running_sum;
}

__global__ void assignDotProduct(const float* dot_products, const float* norms, int* assignments, int num_vecs, int k) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_id >= num_vecs) return;

    float best_score = FLT_MAX;
    int best_k = -1;

    for (int centroidIndex = 0; centroidIndex < k; centroidIndex++) {
        float dot = dot_products[data_id * k + centroidIndex];

        float score = norms[centroidIndex] - 2.0f * dot;

        if (score < best_score) {
            best_score = score;
            best_k = centroidIndex;
        }
    }

    assignments[data_id] = best_k;
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

    float* data = malloc(total_size);

    float* gpu_data;
    cudaMalloc(&gpu_data, total_size);

    int centroid_products_size = sizeof (float) * k)
    int dot_products_size = sizeof (float) * k * num_data;)

    float* gpu_centroid_products;
    cudaMalloc(&gpu_centroid_products, centroid_products_size);

    float* gpu_dot_products;
    cudaMalloc(&gpu_dot_products, dot_products_size);

    // Launch kernels
    cudaMemcpy(gpu_data, data, total_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (k + threads - 1) / threads;
    dotProductCentroids<<<blocks, threads>>>(gpu_centroid_products, k, dim, centroid_products);

    dim3 blockDim(16, 16);
    dim3 gridDim((num_vecs + blockDim.x - 1) / blockDim.x, (k + blockDim.y - 1) / blockDim.y);
    dotProduct<<<gridDim, blockDim>>>(gpu_data, centroids, gpu_dot_products, num_vecs, k, dim);

    int threads = 256;
    int blocks = (num_vecs + threads - 1) / threads;
    assignDotProduct<<<blocks, threads>>>(gpu_dot_products, gpu_centroid_products, assignments, num_vecs, k);

    cudaMemcpy(data, gpu_data, total_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_data);
    cudaFree(gpu_centroid_products);
    cudaFree(gpu_dot_products);

    free(data);

    return 0;
}