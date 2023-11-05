#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

const char* array_fill_name = "array_fill";
const char* sort_check_name = "sort_check";

using namespace std;

void parallel_array_fill(int NUM_VALS, float *values, int num_procs, int rank)
{
    CALI_MARK_BEGIN(array_fill_name);
    
    // Calculate local size based on rank and array size
    int local_size = NUM_VALS / num_procs;
    int start = rank * local_size;
    int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

    local_size = end - start;

    // Print process segment of array
    //printf("start: %d, end: %d, local_size:%d\n", start, end, local_size);

    float *local_values = (float *)malloc(local_size * sizeof(float));

//    for (int i = 0; i < local_size; ++i) 
//    {
//        local_values[i] = (float)rand() / (float)RAND_MAX;
//        printf("Check value for %d : %.6f\n",i, local_values[i]);
//    }

    // Gather local portions into global array
    MPI_Gather(local_values, local_size, MPI_FLOAT, values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(local_values);

    CALI_MARK_END(array_fill_name);
}

bool sort_check(float *local_values, int local_size)
{
    for (int i = 1; i < local_size; i++)
    {
        if (local_values[i - 1] > local_values[i]) 
        {
            return false;
        }
    }
    return true;
}

void parallel_sort_check(int NUM_VALS, float *values, int num_procs, int rank)
{
    CALI_MARK_BEGIN(sort_check_name);

    // Calculate local size based on rank and array size
    int local_size = NUM_VALS / num_procs;
    int start = rank * local_size;
    int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

    local_size = end - start;

    float* local_values = (float*)malloc(local_size * sizeof(float));

    // Scatter the array among processes
    MPI_Scatter(values, local_size, MPI_FLOAT, local_values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Print process segment of array
    printf("start: %d, end: %d, local_size:%d\n", start, end, local_size);

    bool local_sorted = sort_check(local_values, local_size);

    // Gather local portions into global array
    bool all_sorted;
    MPI_Allreduce(&local_sorted, &all_sorted, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        if (all_sorted) 
        {
            // Check if each segment of values is sorted
            float cur_largest = values[local_size - 1];
            for (int i = 1; i < NUM_VALS/local_size; i++)
            {
                if (values[i*local_size] > cur_largest)
                {
                    cur_largest = values[(i+1)*local_size - 1];
                }
                else
                {
                    all_sorted = false;
                    printf("The entire array is not sorted.");
                    break;
                }
            }
            printf("The entire array is sorted.\n");
        }
        else
        {
            printf("The entire array is not sorted.\n");
        }
    }

    free(local_values);

    CALI_MARK_END(sort_check_name);
}

//merge two arrays so that if the two arrays were sorted, the returning array will be sorted
void merge(float *values, int left, int mid, int right)
{
  //printf("merge()%d%d%d", left, mid, right);
  int leftArrayLen = mid - left + 1;
  int rightArrayLen = right - mid;

  float leftArray[leftArrayLen];
  float rightArray[rightArrayLen];

  for(int i = 0; i < leftArrayLen; i++) {
    leftArray[i] = values[left+i];
  }
  for(int i = 0; i < rightArrayLen; i++) {
    rightArray[i] = values[mid + 1 + i];
  }

  int i, j, k;
  i = 0; j = 0;
  k = left;

  while( i < leftArrayLen && j < rightArrayLen) {
    if(leftArray[i] <= rightArray[j]) {
      values[k] = leftArray[i];
      i++;
    } else {
      values[k] = rightArray[j];
      j++;
    }
    k++;
  }

  //finish with whatever's left in either array
  while(i < leftArrayLen) {
    values[k] = leftArray[i];
    i++;
    k++;
  }

  while(j < rightArrayLen) {
    values[k] = rightArray[j];
    j++;
    k++;
  }
}

void mergesort(float *values, int left, int right)
{
  if(left < right) {
    int mid = left + (right - left)/2;
    mergesort(values, left, mid);
    mergesort(values, mid + 1, right);
    merge(values, left, mid, right);
  }
}

void combineArrays(float *a, int lenA, float *b, int lenB, float *c, int &lenC)
{
  lenC = lenA + lenB;
  c = (float *)malloc(lenC * sizeof(float));
  for(int i = 0; i < lenA; i++) {
    c[i] = a[i];
    printf("Adding to c... %.6f\n", a[i]);
    printf("c[i]... %.6f\n", c[i]);
  }
  for(int i = 0; i < lenB; i++) {
    c[i+lenA] = b[i];
    printf("Adding to c... %.6f\n", b[i]);
    printf("c[i+lenA]... %.6f\n", c[i+lenA]);
  }
}


//This was a function I wrote to test all of the functionality for a full recursive multi-processor
//implementation of mergesort
//It works if you do 14 vals and 2 processors
void messagePassingCheck(int NUM_VALS, float *values, int num_procs)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Calculate local size based on rank and array size
  int local_size = NUM_VALS / num_procs;
  int start = rank * local_size;
  int end = (rank == num_procs - 1) ? NUM_VALS : start + local_size;

  local_size = end - start;

  float* local_values = (float*)malloc(local_size * sizeof(float));

  if(rank == 0) {
    printf("0: Check global values...\n");
    for(int i = 0; i < NUM_VALS; i++) {
      printf("0: Value: %.6f\n", values[i]);
    }
  }

  // Scatter the array among processors
  MPI_Scatter(values, local_size, MPI_FLOAT, local_values, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  //build a binary tree?
  
  //First, I want to do some testing with communicating between processes
  //Let's get a process to send another process a message

  if(rank == 0) {
    mergesort(local_values, 0, local_size-1);
    printf("0: I'll be the sender\n");
    printf("0: Sending my values...\n");
    for(int i = 0; i < local_size; i++) {
      printf("0: Value: %.6f\n", local_values[i]);
    }

    MPI_Send(local_values, local_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    printf("0: I sent it!\n");

  }
  if(rank == 1) {
    printf("1: I'll be the receiver\n");
    mergesort(local_values, 0, local_size-1);
    printf("1: Here are my values...\n");
    for(int i = 0; i < local_size; i++) {
      printf("1: Value: %.6f\n", local_values[i]);
    }
    float* foreign_values = (float*)malloc(local_size * sizeof(float));
    MPI_Recv(foreign_values, local_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("1: Here are his values.\n");

    for(int i = 0; i < local_size; i++) {
      printf("1: Value: %.6f\n", foreign_values[i]);
    }
    
    //combineArrays(foreign_values, local_size, local_values, local_size, combined_values, unused);
    
    float* combined_values = new float[local_size * 2];
    std::copy(local_values, local_values + local_size, combined_values);
    std::copy(foreign_values, foreign_values + local_size, combined_values + local_size);
    

    printf("1: Let's join the arrays together\n");

    for(int i = 0; i < NUM_VALS; i++) {
      printf("1: Value: %.6f\n", combined_values[i]);
    }

    merge(combined_values, 0, local_size-1, (local_size * 2)-1);

    printf("1: Okay, are they sorted now?\n");

    for(int i = 0; i < NUM_VALS; i++) {
      printf("1: Value: %.6f\n", combined_values[i]);
    }

    bool result = sort_check(combined_values, NUM_VALS);
    if(result) {
      printf("The array is sorted.\n");
    } else {
      printf("The array is NOT sorted.\n");

    }
   
  }
}

struct node
{
  int value;
  struct node *left;
  struct node *right;
};

void printTree(std::string prefix, node* n, bool isLeft) {
  if(n != nullptr)
  {
    printf(prefix.c_str());
    if(isLeft) {
      printf("├──");
    } else {
      printf("└──");
    }
    printf("%.6f\n",n->value);
    if(isLeft) {
      printTree(prefix + "|   ", n->left, true);
      printTree(prefix + "|   ", n->right, false);
    } else {
      printTree(prefix + "    ", n->left, true);
      printTree(prefix + "    ", n->right, false);
    }
  }
}

void populateNode(node* a, int height, int maxheight, vector<int>* nodebank)
{
  printf("populateNode %d/%d\n", height, maxheight);
  //left node has the same value
  printf("Made a left node\n");
  node* left = new node;
  left->value = a->value;
  a->left = left;
  if(height < maxheight) {
    populateNode(left, height+1, maxheight, nodebank);
  }

  node* right = new node;
  right->value = nodebank->back();
  nodebank->pop_back();
  printf("Popped a node from the nodebank for a right node!\n");
  printf("That node was %d\n", right->value);
  a->right = right;

  if(height < maxheight) {
    populateNode(right, height+1, maxheight, nodebank);
  }
}

void parallelMergesort(int NUM_VALS, float *values, int num_procs)
{ 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //generate a binary tree with num_procs leaves
  //it's height is equal to log_2(num_procs)
  if(rank == 0) {
    node* root = new node;
    root->value = 0;
    int treesHeight = log2(num_procs) + 1;
    printf("building a binary tree\n", treesHeight);
    
    vector<int> *nodebank = new vector<int>(num_procs); //a vector of integers of nodes which we can add
    for(int i = 1; i < num_procs; i++) {
      printf("Does this line crash?\n");
      nodebank->push_back(i);
      printf("Nodebank has %d\n",i);
    }

    populateNode(root, 1, treesHeight, nodebank);

    //check tree
    printTree("", root, false);
  }
}



int main(int argc, char* argv[]) 
{
    srand(time(NULL));
    CALI_CXX_MARK_FUNCTION;

    int NUM_VALS = atoi(argv[1]);

    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    {
      // End execution if either:
      // - num_procs is not a power of two
      // - NUM_VALS is not divisble by num_procs
      int a = num_procs;
      while(a % 2 ==0) {
        a = a / 2;
      }

      if(a == 1) {
        //good
      } else {
        printf("Error: Number of processes isn't a power of two.\n");
        printf("Values: %d", NUM_VALS);
        printf("Procs: %d", num_procs);
        return 1;
      }

      if(NUM_VALS % num_procs == 0) {
        //good
      } else {
        printf("Error: Number of values isn't divisible by number of processes.\n");
        printf("Values: %d", NUM_VALS);
        printf("Procs: %d", num_procs);
        return 1;
      }
    }

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Initialize array to be sorted 
    float *values = (float *)malloc(NUM_VALS * sizeof(float));

    // Fill the local portions of the array then gather into values (NUM_VALS MUST BE DIVISIBLE BY num_procs)
    parallel_array_fill(NUM_VALS, values, num_procs, rank);

    
    //messagePassingCheck(NUM_VALS, values, num_procs);
    parallelMergesort(NUM_VALS, values, num_procs);


    // Check if values is sorted

    free(values);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}
