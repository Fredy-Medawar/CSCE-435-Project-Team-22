#include "common.h"

__global__ void oddeven_dev(float* data, unsigned int N, bool odd) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(odd && (idx%2==0))
        return;
    if(!odd && (idx)%2!=0)
        return;
    
    unsigned int nxt = idx+1;
    if(nxt>=N)
        return;

    if(data[idx]>data[nxt]) {
        float tmp = data[idx];
        data[idx] = data[nxt];
        data[nxt] = tmp;
    }
}

void oddeven(float* values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS) {
    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    CALI_MARK_BEGIN(COMP);
    CALI_MARK_BEGIN(COMP_LARGE);

    for(int i=0;i<NUM_VALS;i++) {
        bool isOdd = i%2!=0;
        oddeven_dev<<<blocks, threads>>>(dev_values, NUM_VALS, isOdd);
    }

    CALI_MARK_END(COMP_LARGE);
    CALI_MARK_END(COMP);
}


