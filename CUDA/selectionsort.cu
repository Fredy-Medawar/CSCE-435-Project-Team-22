#include "common.h"

__global__ void selection_sort_step(float *dev_values, unsigned int* partitionBegin, unsigned int* partitionEnd)
{
    unsigned int start = partitionBegin[threadIdx.x];
    unsigned int end = partitionEnd[start];

    if (start >= end)
        return;

    for (int i = start; i < end; i++) {
        int minIndex = i;
        for (int j = i + 1; j < end; j++) {
            if (dev_values[j] < dev_values[minIndex]) {
                minIndex = j;
            }
        }
        int tmp = dev_values[i];
        dev_values[i] = dev_values[minIndex];
        dev_values[minIndex] = tmp;
    }
}

void selection_sort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS)
{
    dim3 blocks(BLOCKS, 1);    /* Number of blocks */
    dim3 threads(THREADS, 1);  /* Number of threads */
    unsigned int* partitionEnd;
    unsigned int* partitionBegin;
    size_t partsize = NUM_VALS * sizeof(unsigned int);
    cudaMalloc((void**) &partitionEnd, partsize);
    cudaMalloc((void**) &partitionBegin, partsize);
    cudaMemset(partitionEnd, 0, partsize);
    cudaMemset(partitionBegin, NUM_VALS, partsize);

    CALI_MARK_BEGIN(selection_sort_region);
    int i = 0;
    while (i <= NUM_VALS) {
        selection_sort_step<<<blocks, threads>>>(dev_values, partitionBegin, partitionEnd);
        cudaDeviceSynchronize();
        i++;
    }
    cudaDeviceSynchronize();
    cudaFree(partitionEnd);
    cudaFree(partitionBegin);
    CALI_MARK_END(selection_sort_region);
}
