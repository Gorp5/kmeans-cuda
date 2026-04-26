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
__global__ void assignCentroid(ui* assignments, float* data, int num_vecs, float* centroids, ui k, int dim) {
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_id >= num_vecs) return;

    float min_distance = FLT_MAX;
    int min_k = -1;

    float* target_vector = data + data_id * dim;

    
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

    assignments[data_id] = (ui)min_k;
}

__global__ void addAllAssignments(ui* assignments, float* assignment_sums, int* counts, float* data, int k, int dim, int num_vecs){
    int data_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(data_id >= num_vecs){
        return;
    }

    int assigned_cen = (int)assignments[data_id];

    float* data_vals = data + data_id * dim;

    for(int i = 0; i < dim; i++){
        atomicAdd(&assignment_sums[assigned_cen * dim + i], data_vals[i]);
    }

    atomicAdd(&counts[assigned_cen], 1);
}


__global__ void divideSums(float* assignment_sums, int* counts, float* centroids, int k, int dim){
    int centroid_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(centroid_id >= k){
        return;
    }

    if(counts[centroid_id] == 0){
        return;
    }

    float* centroid_out = centroids + centroid_id * dim;
    float* centroid_sum = assignment_sums + centroid_id * dim;

    for(int i = 0; i < dim; i++){
        centroid_out[i] = centroid_sum[i]/(float)counts[centroid_id];
    }


}
//Main kmeans function that calls the three kernels to run kmeans, updates the centroid and assignment values
void kmeans(float* h_data, float* h_centroids, ui* h_assignments, int N, int k, int dim, int max_iterations){
    float *d_data, *d_centroids, *d_sums;
    int *d_counts;
    ui *d_assignments;

    cudaMalloc(&d_data, N * dim * sizeof(float));
    cudaMalloc(&d_centroids, k * dim * sizeof(float));
    cudaMalloc(&d_sums, k * dim * sizeof(float));
    cudaMalloc(&d_assignments, N * sizeof(ui));
    cudaMalloc(&d_counts, k * sizeof(int));

    cudaMemcpy(d_data,      h_data,      N * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, k * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_assignments, 0, N * sizeof(ui));

    int ThrPerBlk     = 256;
    int blocksN = (N + ThrPerBlk - 1) / ThrPerBlk;
    int blocksK = (k + ThrPerBlk - 1) / ThrPerBlk;

    for(int i = 0; i < max_iterations; i++){

        assignCentroid<<<blocksN, ThrPerBlk>>>(d_assignments, d_data, N,
                                         d_centroids, (ui)k, dim);

        cudaMemset(d_sums,   0, k * dim * sizeof(float));
        cudaMemset(d_counts, 0, k       * sizeof(int));

        addAllAssignments<<<blocksN, ThrPerBlk>>>(d_assignments, d_sums, d_counts,
                                            d_data, k, dim, N);

        divideSums<<<blocksK, ThrPerBlk>>>(d_sums, d_counts, d_centroids, k, dim);
    }

    cudaMemcpy(h_assignments, d_assignments, N * sizeof(ui), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, k * dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_assignments);
    cudaFree(d_counts);
}


int main() {
    int N, num_labels;
    int k        = 10;
    int dim      = 784;
    int max_iterations = 300;

    // load
    float* h_data        = loadMNISTImages("train-images-idx3-ubyte", &N);
    int*   h_labels       = loadMNISTLabels("train-labels-idx1-ubyte", &num_labels);
    float* h_centroids   = (float*)malloc(k * dim * sizeof(float));
    ui*    h_assignments = (ui*)malloc(N * sizeof(ui));

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

    printf("Running Kmeans now for %d iterations\n", max_iterations);
    kmeans(h_data, h_centroids, h_assignments, N, k, dim, max_iterations);

    int counts[10] = {};
    for (int i = 0; i < N; i++){
        counts[h_assignments[i]]++;
    }
    
    printf("\ncluster sizes:\n");
    for (int i = 0; i < k; i++){
        printf("  cluster %d: %d points\n", i, counts[i]);
    }

    free(h_data);
    free(h_labels);
    free(h_centroids);
    free(h_assignments);

    return 0;
}