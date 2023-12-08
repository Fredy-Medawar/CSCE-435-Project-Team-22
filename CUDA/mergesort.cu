#include "common.h"

#include <iostream>
#include <helper_cuda.h>
#include <sys/time.h>

// This implementation comes from https://github.com/kevin-albert/cuda-mergesort
// Thanks!

/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its runnning time
 *  is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 *  running on the GPU may provide a significant boost in performance
 */


// data[], size, threads, blocks, 
void mergesort(float*, int, dim3, dim3);
__global__ void gpu_mergesort(float*, float*, int, int, int, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(float*, float*, int, int, int);


// profiling
int tm();

#define min(a, b) (a < b ? a : b)


bool verbose =0;
void merge_sort(float *dev_values, int NUM_VALS, int THREADS, int BLOCKS) {

    
    float *values = (float*) malloc( NUM_VALS * sizeof(float));

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;


    CALI_MARK_BEGIN(COMP);
    CALI_MARK_BEGIN(COMP_LARGE);
    // merge-sort the data
    mergesort(dev_values, NUM_VALS, threadsPerBlock, blocksPerGrid);

    CALI_MARK_END(COMP_LARGE);
    CALI_MARK_END(COMP);

}

void mergesort(float* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    float* D_data;
    float* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
    tm();
    checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(float)));
    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(float), cudaMemcpyHostToDevice));
    if (verbose) 
        std::cout << "cudaMemcpy list to device: " << tm() << " microseconds\n";
 
    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    float* A = D_data;
    float* B = D_swp;

    float nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        float slices = size / ((nThreads) * width) + 1;

        if (verbose) {
            std::cout << "mergeSort - width: " << width 
                      << ", slices: " << slices 
                      << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        if (verbose)
            std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    tm();
    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(float), cudaMemcpyDeviceToHost));
    if (verbose)
        std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";
    
    
    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(float* source, float* dest, int size, int width, int slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    float start = width*idx*slices, 
         middle, 
         end;

    for (float slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(float* source, float* dest, int start, int middle, int end) {
    int i = start;
    int j = middle;
    for (int k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// read data into a minimal linked list
typedef struct {
    int v;
    void* next;
} LinkNode;


// 
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
//
timeval tStart;
int tm() {
    timeval tEnd;
    gettimeofday(&tEnd, 0);
    int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    tStart = tEnd;
    return t;
}


