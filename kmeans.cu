#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

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
        centroid_out[i] = centroid_sums[i]/(float)counts[centroid_id];
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