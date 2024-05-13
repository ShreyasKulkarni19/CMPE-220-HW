#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>

// This program computes a simple version of matrix-vector multiplication and vector-vector addition
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include<chrono>
using std::cout;
using std::generate;
using std::vector;

__global__ void matrixVectorMul(const int* a, const int* b, int* c, int* d, int N) {
    // Compute each thread's global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform matrix-vector multiplication
    c[idx] = 0;
    for (int i = 0; i < N; ++i) {
        c[idx] += a[idx * N + i] * b[i];
    }

    // Perform vector addition
    c[idx] += d[idx];
}

// Check result on the CPU
void verify_result(vector<int>& a, vector<int>& b, vector<int>& c, vector<int>& d, int N) {
    // For every element...
    for (int i = 0; i < N; ++i) {
        // Calculate the result of matrix-vector multiplication and vector addition
        int tmp = 0;
        for (int j = 0; j < N; ++j) {
            tmp += a[i * N + j] * b[j];
        }
        tmp += d[i];
        // Check against the CPU result
        assert(tmp == c[i]);
    }
}

int main() {
    // Matrix size of 1024 x 1024 and vector size of 1024;
    int N = 1 << 10;

    // Size (in bytes) of matrix and vector
    size_t matrix_bytes = N * N * sizeof(int);
    size_t vector_bytes = N * sizeof(int);

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N);
    vector<int> h_c(N);
    vector<int> h_d(N);

    // Initialize matrices and vectors
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
    generate(h_d.begin(), h_d.end(), []() { return rand() % 100; });

    // Allocate device memory
    int* d_a, * d_b, * d_c, * d_d;
    cudaMalloc(&d_a, matrix_bytes);
    cudaMalloc(&d_b, vector_bytes);
    cudaMalloc(&d_c, vector_bytes);
    cudaMalloc(&d_d, vector_bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d.data(), vector_bytes, cudaMemcpyHostToDevice);

    // Threads per block dimension
    int THREADS = 1024;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Launch kernel
    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing
    matrixVectorMul<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, d_d, N);
    cudaDeviceSynchronize(); // Wait for kernel to finish
    auto end_time = std::chrono::high_resolution_clock::now(); // End timing

    // Calculate execution time in seconds
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double execution_time = elapsed_seconds.count();

    // Count floating point operations (assuming 2 operations per element)
    long long total_flops = 2 * static_cast<long long>(N * N * N);

    // Calculate GFLOPS
    double gflops = total_flops / (execution_time * 1e9);

    // Output GFLOPS
    std::cout << "GFLOPS: " << gflops << std::endl;

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, vector_bytes, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c, h_d, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}
