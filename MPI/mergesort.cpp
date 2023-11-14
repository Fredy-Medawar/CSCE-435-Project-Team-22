#include "common.h"

using namespace std;

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

struct node
{
  int value;
  struct node *left = nullptr;
  struct node *right = nullptr;
  struct node *parent = nullptr;
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
    printf("%d\n",n->value,n);
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
  if(height == maxheight) {return;}
  node* left = new node;
  left->value = a->value;
  left->parent = a;
  a->left = left;
  if(height < maxheight) {
    populateNode(left, height+1, maxheight, nodebank);
  }

  node* right = new node;
  right->value = nodebank->back();
  right->parent = a;
  nodebank->pop_back();
  a->right = right;

  if(height < maxheight) {
    populateNode(right, height+1, maxheight, nodebank);
  }
}

node* searchForLeaf(node* n, int value) {
  if(n == nullptr) {return nullptr;}
  if(n->left != nullptr) {
    node* leftSearchResult = searchForLeaf(n->left, value);
    node* rightSearchResult = searchForLeaf(n->right, value);
    if(leftSearchResult != nullptr) {
      return leftSearchResult;
    } else {
      return rightSearchResult;
    }
  } else {
    if(n->value == value) {
      return n;
    }
  }
  return nullptr;
}

string stepUpTree(node* n, string ret) {
  
  ret += to_string(n->value);
  if(n->parent == nullptr) { 
    return ret; 
  } else {
    return stepUpTree(n->parent, ret);
  }

}

//mergesort(NUM_VALS, values, num_procs, rank);
void mergesort(int NUM_VALS, float* local_values, int local_size, int num_procs, int rank)
{ 
  CALI_MARK_BEGIN(MPI_mergesort);

  //generate a binary tree with num_procs leaves
  //it's height is equal to log_2(num_procs)
  node* root = new node;
  root->value = 0;
  int treesHeight = log2(num_procs) + 1;
  
  vector<int> *nodebank = new vector<int>(num_procs); //a vector of integers of nodes which we can add
  for(int i = 1; i < num_procs; i++) {
    nodebank->push_back(i);
  }

  populateNode(root, 1, treesHeight, nodebank);

  //check tree
  if(rank == 0) {
    printTree("", root, false);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //the tree is a representation of how the threads will pass data.
  
  //search tree for leaf with value == rank
  node* myLeaf = searchForLeaf(root, rank);

  //derive a list of destinations that this thread's data will go to
  string destList = stepUpTree(myLeaf, "");
  //printf("My rank is %d and my dest list is %s.\n", rank, destList.c_str());
  
  destList.erase(0,1);

  int i = 1; 
  bool work = 1; //some processors pass through the loop and don't work
  int current_size = local_size;
  for(;;) {
    
    if(work) {
      mergesort(local_values, 0, current_size-1);
  
      string thisDest = "";
      thisDest += destList[0];
      destList.erase(0,1);
      int dest = stoi(thisDest);
  
      if(dest == rank) {
        //we need to listen for a message from the other processor
        float* foreign_values = (float*)malloc(current_size * sizeof(float));
        MPI_Recv(foreign_values, current_size, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float* combined_values = new float[current_size * 2];
        std::copy(local_values, local_values + current_size, combined_values);
        std::copy(foreign_values, foreign_values + current_size, combined_values + current_size);
        merge(combined_values, 0, current_size-1, (current_size * 2)-1);

        free(local_values);
        free(foreign_values);
        local_values = combined_values;
        
      } else {
        //send it to dest
        MPI_Send(local_values, current_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        work = 0; //this processor now rests
      }
    }

    current_size = current_size * 2;

    MPI_Barrier(MPI_COMM_WORLD);

    if(destList.size() == 0) {break;}
  }
  CALI_MARK_END(MPI_mergesort);

  CALI_MARK_BEGIN(sort_check_name);
  //process 0 now has the final result
  if(rank == 0) {
    for(int i = 0; i < NUM_VALS; i++) {
      printf("0: Value: %.6f\n", local_values[i]);
    }
    bool result = sort_check(local_values, NUM_VALS);
    if(result) {
      printf("The array is sorted.\n");
    } else {
      printf("The array is NOT sorted.\n");
    }
  }
  CALI_MARK_END(sort_check_name);  
}

