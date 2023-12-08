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
    CALI_MARK_BEGIN(COMM);
    CALI_MARK_BEGIN(COMM_LARGE);
    CALI_MARK_BEGIN(cudaMemcpy_host_to_device);

    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sorted, &sorted, sizeof(int), cudaMemcpyHostToDevice);

    // End  host to device Event
    cudaDeviceSynchronize();
    CALI_MARK_END(cudaMemcpy_host_to_device);
    CALI_MARK_END(COMM_LARGE);
    CALI_MARK_END(COMM);

    // Start Array Fill Event
    CALI_MARK_BEGIN(array_fill_name);
    srand(time(NULL));
    unsigned long long seed = (unsigned long long)rand();
    device_array_fill<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS, seed, array_fill_type);

    // End Array Fill Event
    cudaDeviceSynchronize();
    CALI_MARK_END(array_fill_name);

    if (sort_alg == 0)
    {
        bitonic_sort(values, dev_values, NUM_VALS, THREADS, BLOCKS);
    } else if(sort_alg==1) {
        oddeven(values, dev_values, NUM_VALS, THREADS, BLOCKS);
    } else if(sort_alg==2) {
        selectionsort(values, dev_values, NUM_VALS, THREADS, BLOCKS);
    } else if(sort_alg == 3) {
        merge_sort(dev_values, NUM_VALS, THREADS, BLOCKS);
    }

    CALI_MARK_BEGIN(sort_check_name);
    device_sort_check<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS, dev_sorted);
    cudaDeviceSynchronize();
    CALI_MARK_END(sort_check_name);

    //MEM COPY FROM DEVICE TO HOST

    // Start device to host event
    CALI_MARK_BEGIN(COMM);
    CALI_MARK_BEGIN(COMM_LARGE);
    CALI_MARK_BEGIN(cudaMemcpy_device_to_host);

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&sorted, dev_sorted, sizeof(int), cudaMemcpyDeviceToHost);

    // End device to host Event
    cudaDeviceSynchronize();
    CALI_MARK_END(cudaMemcpy_device_to_host);
    CALI_MARK_END(COMM_LARGE);
    CALI_MARK_END(COMM);

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

    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // Size of input
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 22); // The number of your group (integer, e.g., 1, 10)


    if (array_fill_type == 0) adiak::value("InputType", "random");
    else if (array_fill_type == 1) adiak::value("InputType", "sorted");
    else if (array_fill_type == 2) adiak::value("InputType", "reverse_sorted");
    else if (array_fill_type == 3) adiak::value("InputType", "sorted_1%_perturbed");

    if (sort_alg == 0) adiak::value("Algorithm", "bitonic_sort");
    else if (sort_alg == 1) adiak::value("Algorithm", "oddeven_sort");
    else if (sort_alg == 2) adiak::value("Algorithm", "selection_sort");
    else if (sort_alg == 3) adiak::value("Algorithm", "mergesort");

    //array_print(values, NUM_VALS);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}
