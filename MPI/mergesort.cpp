#include "common.h"

using namespace std;

bool sort_check_c_style(float* local_values, int local_size)
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

void local_mergesort(float *values, int left, int right)
{
  if(left < right) {
    int mid = left + (right - left)/2;
    local_mergesort(values, left, mid);
    local_mergesort(values, mid + 1, right);
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

vector<int> stepUpTree(node* n, vector<int> v) {
  v.push_back(n->value);
  if(n->parent == nullptr) { 
    return v; 
  } else {
    return stepUpTree(n->parent, v);
  }

}

//mergesort(NUM_VALS, values, num_procs, rank);
void mergesort(int NUM_VALS, vector<float> *flocal_values, float* local_arr, int local_size, int num_procs, int rank)
{ 
  //convert vector<float>* to c-style arr
  for(int i = 0; i < local_size; i++) {
    local_arr[i] = (*flocal_values)[i];
  }

  CALI_MARK_BEGIN(COMP);
  CALI_MARK_BEGIN(COMP_LARGE);

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
    //printTree("", root, false);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  //the tree is a representation of how the threads will pass data.
  
  //search tree for leaf with value == rank
  node* myLeaf = searchForLeaf(root, rank);

  //derive a list of destinations that this thread's data will go to
  vector<int> destList;
  destList = stepUpTree(myLeaf, destList);
  
  destList.erase(destList.begin());

  int i = 1; 
  bool work = 1; //some processors pass through the loop and don't work
  int current_size = local_size;
  for(;;) {
    
    if(work) {
      local_mergesort(local_arr, 0, current_size-1);
  
      string thisDest = "";
      int dest = destList[0];
      destList.erase(destList.begin());
  
      if(dest == rank) {
        //we need to listen for a message from the other processor
        float* foreign_values = (float*)malloc(current_size * sizeof(float));
        
        MPI_Recv(foreign_values, current_size, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float* combined_values = new float[current_size * 2];
        std::copy(local_arr, local_arr + current_size, combined_values);
        std::copy(foreign_values, foreign_values + current_size, combined_values + current_size);
        merge(combined_values, 0, current_size-1, (current_size * 2)-1);
        free(local_arr);
        free(foreign_values);
        local_arr = combined_values;
        
        
      } else {
        //send it to dest
        MPI_Send(local_arr, current_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        work = 0; //this processor now rests
      }
    } else {
      destList.erase(destList.begin());
    }

    current_size = current_size * 2;

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(destList.size() == 0) {break;}
  }


  CALI_MARK_END(COMP_LARGE);
  CALI_MARK_END(COMP);

  CALI_MARK_BEGIN(SORT_CHECK_NAME);
  
  if(rank == 0) {
//      for(int i = 0; i < NUM_VALS; i++) {
//          printf("0: Value: %.6f\n", local_arr[i]);
//      }
      bool result = sort_check_c_style(local_arr, NUM_VALS);
      if(result) {
          //printf("The array is sorted Joseph.\n");
      } else {
          //printf("The array is sorted Joseph.\n");
      }
  }

  CALI_MARK_END(SORT_CHECK_NAME);  
}

