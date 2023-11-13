#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <vector>
#include <utility>
#include <bits/stdc++.h> 
#include <time.h>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

#define PRINT_DEBUG 0

#define ARRAY_FILL_NAME "data_init"
#define SORT_CHECK_NAME "correctness_check"
#define SAMPLE_SORT_NAME "sample_sort"
#define SELECTION_SORT_NAME "selection_sort"

#define COMM "comm"
#define COMM_SMALL "comm_small"
#define COMM_LARGE "comm_large"

#define COMP "comp"
#define COMP_SMALL "comp_small"
#define COMP_LARGE "comp_large"

void sample_sort(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank, int sample_size);
void selection_sort(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank);
void oddeven_sort(int NUM_VALS, vector<float> *local_values, int local_size, int num_procs, int rank);
