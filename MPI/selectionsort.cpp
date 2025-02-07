#include "common.h"

float smallest(float* a, int b, int c) {
    int temp = b;
    for (int x = b + 1; x < c; x++) {
        if (a[temp] > a[x])
            temp = x;
    }
    float z = a[temp];
    a[temp] = a[b];
    a[b] = z;

    return a[b];
}

void selection_sort(int NUM_VALS, vector<float>* local_values, int local_size, int num_procs, int rank) {
    CALI_MARK_BEGIN(SELECTION_SORT_NAME);

    float* selectionArrayA = (float*)malloc(NUM_VALS * sizeof(float));

    int localIndex;

    if (rank == 0) {
        srand(time(NULL) + rank);
        printf("This is the unsorted array: ");
        for (int i = 0; i < NUM_VALS; i++) {
            localIndex = rand() % local_size;
            selectionArrayA[i] = local_values->at(localIndex);
        }
    }

    float* selected = NULL;
    float* gatheredArray = NULL; // Separate array for gathering data
    int smallestValue = 0;
    int smallestInProcess;

    if (rank == 0) {
        selected = (float*)malloc(NUM_VALS * sizeof(float));
        gatheredArray = (float*)malloc(NUM_VALS * sizeof(float));
    }

    float* selectionArrayB = (float*)malloc(local_size * sizeof(float));

    // CALI_MARK_BEGIN("Scatter Array");

    MPI_Scatter(selectionArrayA, local_size, MPI_FLOAT, selectionArrayB, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    smallestInProcess = smallest(selectionArrayB, 0, local_size);

    // CALI_MARK_END("Scatter Array");

    int smallestProcess = 0;
    int startPoint = 0;
    int isNull = 0;

    // CALI_MARK_BEGIN("Selection Sort");

    MPI_Barrier(MPI_COMM_WORLD);

    for (int a = 0; a < NUM_VALS; a++) {
        smallestProcess = 0;

        if (rank == 0) {
            if (!isNull)
                smallestValue = smallestInProcess;
            else {
                smallestValue = 150;
            }
        }

        for (int b = 1; b < num_procs; b++) {
            if (rank == 0) {
                int receive;
                MPI_Recv(&receive, 1, MPI_INT, b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (receive != 150) {
                    if (receive < smallestValue) {
                        smallestValue = receive;
                        smallestProcess = b;
                    }
                }
            } else if (rank == b) {
                if (!isNull) {
                    MPI_Send(&smallestInProcess, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                } else {
                    int x = 69; // no useful data
                    MPI_Send(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                }
            }
        }

        MPI_Bcast(&smallestProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            selected[a] = smallestValue;
        }

        if (rank == smallestProcess) {
            startPoint++;
            smallestInProcess = smallest(selectionArrayB, startPoint, local_size);
            if (startPoint > local_size - 1) {
                isNull = 1;
            }
        }
    }

    // CALI_MARK_END("Selection Sort");

    // Gather the sorted values from all processes to process rank 0
    MPI_Gather(selected, NUM_VALS, MPI_FLOAT, gatheredArray, NUM_VALS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nThis is the sorted array: ");
        for (int c = 0; c < NUM_VALS; c++) {
            if (c % num_procs == 0)
                printf("\n");
            printf("%3f ", gatheredArray[c]);
        }
        printf("\n");
        printf("\n");
    }

    free(selectionArrayA);
    free(selectionArrayB);

    if (rank == 0) {
        free(selected);
        free(gatheredArray);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    CALI_MARK_END(SELECTION_SORT_NAME);
}
