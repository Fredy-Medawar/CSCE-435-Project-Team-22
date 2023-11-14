#include "common.h"

//these functions are for a true in-place mergesort
// imsort(), swap(), wmerge(), wsort(), and imsort() are copyrighted under GNU GPL and modified from Liu Xinyu's work (github.com/liuxinyu95/AlgoXY/blob/algoxy/sorting/merge-sort/src/mergesort.c)
__device__ __host__ void imsort(float* xs, int l, int u);

__device__ __host__ void swap(float* xs, int i, int j) {
  float tmp = xs[i]; xs[i] = xs[j]; xs[j] = tmp;
}

__device__ __host__ void wmerge(float* xs, int i, int m, int j, int n, int w) {
  while (i < m && j < n)
    swap(xs, w++, xs[i] < xs[j] ? i++ : j++);
  while (i < m)
    swap(xs, w++, i++);
  while (j < n)
    swap(xs, w++, j++);
} 

__device__ __host__ void wsort(float* xs, int l, int u, int w) {
  int m;
  if (u - l > 1) {
    m = l + (u - l) / 2;
    imsort(xs, l, m);
    imsort(xs, m, u);
    wmerge(xs, l, m, m, u, w);
  }
  else
    while (l < u)
      swap(xs, l++, w++);
}

__device__ __host__ void imsort(float* xs, int l, int u) {
  int m, n, w;
  if (u - l > 1) {
    m = l + (u - l) / 2;
    w = l + u - m;
    wsort(xs, l, m, w); /* the last half contains sorted elements */
    while (w - l > 2) {
      n = w;
      w = l + (n - l + 1) / 2;
      wsort(xs, w, n, l);  /* the first half of the previous working area contains sorted elements */
      wmerge(xs, l, l + n - w, n, u, w);
    }
    for (n = w; n > l; --n) /*switch to insertion sort*/
      for (m = n; m < u && xs[m] < xs[m-1]; ++m)
        swap(xs, m, m - 1);
  }
}

__global__ void mergesortStep(float *values, int num_vals, int sectionWidth)
{
  int id = threadIdx.x; 

  //derive our section of the array based on thread id
  int left = sectionWidth * id;
  int right = left + sectionWidth;
  imsort(values, left, right);
}

void mergesort(float *dev_values, int num_vals, int blocks)
{
  int sliceWidth = 2;
  int threadsToUse = THREADS;

  if(threadsToUse > num_vals / sliceWidth) { threadsToUse = num_vals/sliceWidth;}

  if(threadsToUse < num_vals / sliceWidth) {sliceWidth = num_vals / threadsToUse;}

  for(;;) {
    mergesortStep<<<blocks, threadsToUse>>>(dev_values, num_vals, sliceWidth);
    if(threadsToUse == 1) { break; }
    sliceWidth *= 2;
    threadsToUse /= 2;
  }
}
