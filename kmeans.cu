#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <stdint.h>

typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

uint32_t flipBytes(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8)  |
           ((val & 0x0000FF00) << 8)  |
           ((val & 0x000000FF) << 24);
}

float* loadMNISTImages(const char* filepath, int* num_images) {
    FILE* f = fopen(filepath, "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", filepath); return NULL; }

    uint32_t magic, n_images, n_rows, n_cols;
    fread(&magic,    sizeof(uint32_t), 1, f);
    fread(&n_images, sizeof(uint32_t), 1, f);
    fread(&n_rows,   sizeof(uint32_t), 1, f);
    fread(&n_cols,   sizeof(uint32_t), 1, f);

    magic    = flipBytes(magic);
    n_images = flipBytes(n_images);
    n_rows   = flipBytes(n_rows);
    n_cols   = flipBytes(n_cols);

    if (magic != 2051) { fprintf(stderr, "wrong magic number\n"); return NULL; }

    int dim = n_rows * n_cols;
    *num_images = (int)n_images;

    uint8_t* raw  = (uint8_t*)malloc(n_images * dim);
    fread(raw, sizeof(uint8_t), n_images * dim, f);
    fclose(f);

    float* data = (float*)malloc(n_images * dim * sizeof(float));
    for (int i = 0; i < (int)(n_images * dim); i++)
        data[i] = raw[i] / 255.0f;

    free(raw);
    return data;
}

int* loadMNISTLabels(const char* filepath, int* num_labels) {
    FILE* f = fopen(filepath, "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", filepath); return NULL; }

    uint32_t magic, n_labels;
    fread(&magic,    sizeof(uint32_t), 1, f);
    fread(&n_labels, sizeof(uint32_t), 1, f);

    magic    = flipBytes(magic);
    n_labels = flipBytes(n_labels);

    if (magic != 2049) { fprintf(stderr, "wrong magic number\n"); return NULL; }

    *num_labels = (int)n_labels;

    uint8_t* raw = (uint8_t*)malloc(n_labels);
    fread(raw, sizeof(uint8_t), n_labels, f);
    fclose(f);

    int* labels = (int*)malloc(n_labels * sizeof(int));
    for (int i = 0; i < (int)n_labels; i++)
        labels[i] = (int)raw[i];

    free(raw);
    return labels;
}


void initCentroids(float* h_data, float* centroids, int num_data, int numCentroids, int dim){
    float* distances = (float*)malloc(num_data * sizeof(float));

    //pick first centroid completely at random
    int first = rand() % num_data;
    memcpy(centroids, h_data + first * dim, dim * sizeof(float));

    for (int i = 1; i < numCentroids; i++) {
        // compute each point's distance to its nearest already-chosen centroid
        float total = 0.0f;
        for (int p = 0; p < num_data; p++) {
            float min_dist = FLT_MAX;

            for (int c = 0; c < i; c++) {
                float dist = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = h_data[p * dim + d] - centroids[c * dim + d];
                    dist += diff * diff;
                }
                if (dist < min_dist) min_dist = dist;
            }

            distances[p] = min_dist;
            total += min_dist;
        }

        //pick next centroid with probability proportional to distance
        float threshold = ((float)rand() / RAND_MAX) * total;
        float running = 0.0f;
        int chosen = 0;
        for (int p = 0; p < num_data; p++) {
            running += distances[p];
            if (running >= threshold) {
                chosen = p;
                break;
            }
        }

        memcpy(centroids + i * dim, h_data + chosen * dim, dim * sizeof(float));
    }

    free(distances);
}

// convert to using the nice cuda types later
// Dot product optimization can be done using: ||x−c||^2 = ||x||^2 + ||c||^2 − 2(x * c)
__global__ void assignCentroid(unsigned int* assignments, float* data, int num_vecs, float* centroids, int k, int data_size) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_id >= num_vecs) return;

    float min_distance = FLT_MAX;
    int min_k = -1;

    float* target_vector = data + data_id * data_size;

    for (int current_k = 0; current_k < k; current_k++) {

        // Distance Operation
        float running_sum = 0;
        for (int current_dim = 0; current_dim < data_size; current_dim++) {
            float distance = target_vector[current_dim] - centroids[current_k * data_size + current_dim];
            running_sum += distance * distance;
        }

        if (running_sum < min_distance) {
            min_distance = running_sum;
            min_k = current_k;
        }
    }

    assignments[data_id] = min_k;
}

__global__ void dotProductData(float* data, float* centroids, float* results, int num_vecs, int k, int data_size) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;
    int target_k = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_id >= num_vecs || target_k >= k) return;

    const float* x = &data[data_id * data_size];
    const float* c = &centroids[target_k * data_size];

    float running_sum = 0;

    for (int index = 0; index < data_size; index++) {
        running_sum += x[index] * c[index];
    }

    results[data_id * k + target_k] = running_sum;
}

__global__ void dotProductCentroids(float* centroids, int k, int data_size, float* results) {
    int target_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (target_k >= k) return;

    const float* c = &centroids[target_k * data_size];

    float running_sum = 0;

    for (int index = 0; index < data_size; index++) {
        running_sum += c[index] * c[index];
    }

    results[target_k] = running_sum;
}

__global__ void assignDotProduct(const float* dot_products, const float* norms, unsigned int* assignments, int num_vecs, int k) {
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
    int N, num_labels;
    int k        = 10;
    int dim      = 784;
    int max_iter = 300;

    // load
    float* h_data        = loadMNISTImages("train-images-idx3-ubyte", &N);
    int*   h_labels       = loadMNISTLabels("train-labels-idx1-ubyte", &num_labels);
    float* h_centroids   = (float*)malloc(k * dim * sizeof(float));
    unsigned int*    h_assignments = (unsigned int*)malloc(N * sizeof(unsigned int));

    if (!h_data || !h_labels){
        return 1;
    }

    initCentroids(h_data, h_centroids, N, k, dim);

    //Check distances between Centroids after intialization
    printf("\npairwise centroid distances:\n");
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float delta = h_centroids[i * dim + d] - h_centroids[j * dim + d];
                dist += delta * delta;
            }
            printf("  centroid %d <-> centroid %d  dist²=%.2f\n", i, j, dist);
        }
    }

    size_t total_size = N * dim * sizeof(float);

    float* gpu_data;
    cudaMalloc(&gpu_data, total_size);

    int centroid_products_size = sizeof(float) * k;
    int dot_products_size = sizeof(float) * k * N;
    int centroids_size = sizeof(float) * k * dim;
    int assignments_size = sizeof(unsigned int) * N;

    float* gpu_centroid_products;
    cudaMalloc(&gpu_centroid_products, centroid_products_size);

    float* gpu_dot_products;
    cudaMalloc(&gpu_dot_products, dot_products_size);

    float* gpu_centroids;
    cudaMalloc(&gpu_centroids, centroids_size);

    float* gpu_assignments;
    cudaMalloc(&gpu_assignments, assignments_size);

    // Launch kernels
    cudaMemcpy(gpu_data, h_data, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_assignments, h_assignments, assignments_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (k + threads - 1) / threads;
    dotProductCentroids<<<blocks, threads>>>(gpu_centroids, k, dim, gpu_centroid_products);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (k + blockDim.y - 1) / blockDim.y);
    dotProductData<<<gridDim, blockDim>>>(gpu_data, gpu_centroids, gpu_dot_products, N, k, dim);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    threads = 256;
    blocks = (N + threads - 1) / threads;
    assignDotProduct<<<blocks, threads>>>(gpu_dot_products, gpu_centroid_products, gpu_assignments, N, k);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, gpu_data, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, gpu_centroids, centroids_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_data);
    cudaFree(gpu_centroid_products);
    cudaFree(gpu_dot_products);
    cudaFree(gpu_centroids);
    cudaFree(gpu_assignments);


    free(h_data);
    free(h_labels);
    free(h_centroids);
    free(h_assignments);

    return 0;
}