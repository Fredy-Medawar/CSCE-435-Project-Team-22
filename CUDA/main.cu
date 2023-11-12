#include "common.h"

#include <curand_kernel.h>

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%f ",  arr[i]);
  }
  printf("\n");
}

__global__ void device_array_fill(float *arr, int length, unsigned long long seed, int array_fill_type)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);

    if (i < length)
    {
        if (array_fill_type == 0)
        {
            arr[i] = curand_uniform(&state);
        }
        else if (array_fill_type == 1)
        {
            arr[i] = (float)i;
        }
        else if (array_fill_type == 2)
        {
            arr[i] = (float)(length - i);
        }
        else if (array_fill_type == 3)
        {
            int val = (int)(curand_uniform(&state) * 100) + 1;
            if (val > 1)
            {
                arr[i] = (float)i;
            }
            else
            {
                arr[i] = curand_uniform(&state);
            }
        }
    }
}

__global__ void device_sort_check(const float *arr, int length, int *sorted) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < length - 1) 
    {
        if (arr[i] > arr[i + 1]) 
        {
            atomicExch(sorted, 0);
        }
    }
}

int main(int argc, char *argv[])
{
    int NUM_VALS = atoi(argv[1]);
    int THREADS = atoi(argv[2]);
    int array_fill_type = atoi(argv[3]);
    int sort_alg = atoi(argv[4]);

    int BLOCKS = NUM_VALS / THREADS;

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    // Allocate array on device
    float *values = (float*) malloc( NUM_VALS * sizeof(float));
    float *dev_values;
    size_t size = NUM_VALS * sizeof(float);
    cudaMalloc((void**) &dev_values, size);

    // Allocate sorted flag on device
    int sorted = 1;
    int* dev_sorted; // Device pointer
    cudaMalloc((void**)&dev_sorted, sizeof(int));

    //MEM COPY FROM HOST TO DEVICE

    // Start host to device event
    CALI_MARK_BEGIN(cudaMemcpy_host_to_device);

    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sorted, &sorted, sizeof(int), cudaMemcpyHostToDevice);

    // End  host to device Event
    cudaDeviceSynchronize();
    CALI_MARK_END(cudaMemcpy_host_to_device);

    // Start Array Fill Event
    CALI_MARK_BEGIN(array_fill_name);
    srand(time(NULL));
    unsigned long long seed = 10; // (unsigned long long)rand()
    device_array_fill<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS, seed, array_fill_type);

    // End Array Fill Event
    cudaDeviceSynchronize();
    CALI_MARK_END(array_fill_name);

    if (sort_alg == 0)
    {
        bitonic_sort(values, dev_values, NUM_VALS, THREADS, BLOCKS);
    } else if(sort_alg==1) {
        // oddeven_sort(values, dev_values, NUM_VALS, THREADS, BLOCKS);
    } else if(sort_alg==2) {
        selectionsort(values, dev_values, NUM_VALS, THREADS, BLOCKS);
    }

    CALI_MARK_BEGIN(sort_check_name);
    device_sort_check<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS, dev_sorted);
    cudaDeviceSynchronize();
    CALI_MARK_END(sort_check_name);

    //MEM COPY FROM DEVICE TO HOST

    // Start device to host event
    CALI_MARK_BEGIN(cudaMemcpy_device_to_host);

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&sorted, dev_sorted, sizeof(int), cudaMemcpyDeviceToHost);

    // End device to host Event
    cudaDeviceSynchronize();
    CALI_MARK_END(cudaMemcpy_device_to_host);

    cudaFree(dev_values);

    if (sorted)
    {
        printf("Array is sorted\n");
    }
    else
    {
        printf("Array is not sorted\n");
    }

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("datatype_size", sizeof(float));

    if (array_fill_type == 0) adiak::value("array_fill_type", "random");
    else if (array_fill_type == 1) adiak::value("array_fill_type", "sorted");
    else if (array_fill_type == 2) adiak::value("array_fill_type", "reverse_sorted");
    else if (array_fill_type == 3) adiak::value("array_fill_type", "sorted_1%_perturbed");

    if (sort_alg == 0) adiak::value("sort_alg", "bitonic_sort");
    else if (sort_alg == 1) adiak::value("sort_alg", "oddeven_sort");
    else if (sort_alg == 2) adiak::value("sort_alg", "selection_sort");

    array_print(values, NUM_VALS);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}