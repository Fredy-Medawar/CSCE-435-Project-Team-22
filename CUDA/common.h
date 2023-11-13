#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define bitonic_sort_step_region "bitonic_sort_step"
#define selection_sort_region "selection_sort_region"
#define cudaMemcpy_host_to_device "cudaMemcpy"
#define cudaMemcpy_device_to_host "cudaMemcpy"
#define array_fill_name "data_init"
#define sort_check_name "correctness_check"

#define COMM "comm"
#define COMM_SMALL "comm_small"
#define COMM_LARGE "comm_large"

#define COMP "comp"
#define COMP_SMALL "comp_small"
#define COMP_LARGE "comp_large"

void bitonic_sort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS);
void quicksort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS);
void selectionsort(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS);
void oddeven(float *values, float* dev_values, int NUM_VALS, int THREADS, int BLOCKS);