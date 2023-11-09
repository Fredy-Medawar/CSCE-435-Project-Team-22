# CSCE 435 Group project

## 0. Group number: 22

## 1. Group members:
1. Joseph Buskmiller
2. Fredy Medawar
3. Shreeman Kuppa Jayaram
4. Ahsan Yahya

Communication will be by discord.

---

## 2. _due 10/25_ Project topic
Performance of different implementations of different sorting algorithms in MPI and CUDA

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Sample Sort (MPI)
- Bitonic Sort (CUDA)
- Odd even sort (CUDA, MPI)
- Mergesort (CUDA, MPI)
- Selection Sort (CUDA, MPI)

## 2. Pseudocode

### Selection Sort (MPI):
    function smallest(a: array of float, b: int, c: int) -> float
    Create a variable temp and set it to b
    Iterate from x in range b + 1 to c - 1
        If a[temp] is greater than a[x]
            Update temp to x
        End If
    End Iterate
    Create a variable z and set it to a[temp]
    Swap a[temp] and a[b]
    Return a[b]
    End function

    function selection_sort(NUM_VALS: int, local_values: array of float, local_size: int, num_procs: int, rank: int)
        Allocate memory for selectionArrayA as an array of NUM_VALS floats
        Initialize localIndex to 0
    
        If rank is 0
            Seed the random number generator with the current time plus rank
            Print "This is the unsorted array: "
            For i in range 0 to NUM_VALS - 1
                Generate a random localIndex within the range local_size
                Set selectionArrayA[i] to local_values[localIndex]
            End For
        End If
    
        Initialize selected and gatheredArray to NULL
        Initialize smallestValue to 0 and smallestInProcess to 0
    
        If rank is 0
            Allocate memory for selected as an array of NUM_VALS floats
            Allocate memory for gatheredArray as an array of NUM_VALS floats
        End If
    
        Allocate memory for selectionArrayB as an array of local_size floats
    
        Use MPI to scatter selectionArrayA to selectionArrayB
    
        Set smallestInProcess to the result of smallest(selectionArrayB, 0, local_size)
    
        Initialize smallestProcess to 0
        Initialize startPoint to 0
        Initialize isNull to 0
    
        For a in range 0 to NUM_VALS - 1
            Set smallestProcess to 0
    
            If rank is 0
                If isNull is false
                    Set smallestValue to smallestInProcess
                Else
                    Set smallestValue to 150
                End If
            End If
    
            For b in range 1 to num_procs - 1
                If rank is 0
                    Receive receive from process b using MPI
                    If receive is not 150 and receive is less than smallestValue
                        Set smallestValue to receive
                        Set smallestProcess to b
                    End If
                ElseIf rank is b
                    If isNull is false
                        Send smallestInProcess to process 0 using MPI
                    Else
                        Set x to 69  # No useful data
                        Send x to process 0 using MPI
                    End If
                End If
            End For
    
            Broadcast smallestProcess using MPI
            Use MPI Barrier
    
            If rank is 0
                Set selected[a] to smallestValue
            End If
    
            If rank is smallestProcess
                Increment startPoint by 1
                Set smallestInProcess to the result of smallest(selectionArrayB, startPoint, local_size)
                If startPoint is greater than local_size - 1
                    Set isNull to 1
                End If
            End For
        End For
    
        Use MPI to gather selected into gatheredArray
    
        If rank is 0
            Print "\nThis is the sorted array: "
            For c in range 0 to NUM_VALS - 1
                If c modulo num_procs is 0
                    Print "\n"
                Print gatheredArray[c]
            End For
            Print "\n\n"
        End If
    
        Free memory for selectionArrayA and selectionArrayB
    
        If rank is 0
            Free memory for selected and gatheredArray
        End If
    
        Use MPI Barrier
        Use MPI Finalize
    End function

### Selection Sort (CUDA)
    selection_sort_step(dev_values, partitionBegin, partitionEnd)
    
        Get the thread index threadIdx.x
    
        Initialize start with partitionBegin[threadIdx.x]
        Initialize end with partitionEnd[start]
    
        If start is greater than or equal to end then Return
    
        For i from start to end - 1 do
            Initialize minIndex with i
    
            For j from i + 1 to end - 1 do
                If dev_values[j] is less than dev_values[minIndex] then
                    Set minIndex to j
    
            Swap dev_values[i] and dev_values[minIndex]
    End Function
    
    Function selectionsort(values: host float array, dev_values: device float array, NUM_VALS: int, THREADS: int, BLOCKS: int)
    
        Initialize the number of blocks with BLOCKS
        Initialize the number of threads per block with THREADS
        Initialize partitionEnd as a device unsigned int array of size NUM_VALS
        Initialize partitionBegin as a device unsigned int array of size NUM_VALS
    
        Allocate device memory for partitionEnd and partitionBegin
    
        Initialize partsize with NUM_VALS * size of unsigned int
        Set every element in partitionEnd and partitionBegin to 0 and NUM_VALS in device memory
    
        Initialize i to 0
    
        While i is less than NUM_VALS
            Launch the selection_sort_step kernel with blocks, threads, dev_values, partitionBegin, and partitionEnd
    
            Synchronize the device to ensure the kernel completes
    
            Increment i
    
        Synchronize the device again to ensure all kernels are complete
    
        Free device memory for partitionEnd and partitionBegin
    
    End Function





### Mergesort
    merge(arr, start, mid, end)
      treat start-mid and mid+1 - end as seperate lists
      parse through the lists putting the smaller value first
      after parsing is completely, fill the rest of the array with any leftover values
      
    mergesort(arr, start, end):
        if start > end 
            return
        set mid = (start+end)/2
        mergeSort(arr, start, mid)
        mergeSort(arr, mid+1, end)
        merge(arr, start, mid, end)

### Samplesort
    sample_sort(size, local values, local size, num procs, sample_size)
      create list local values list on each process

      create list sample of size sample_size

      find splitters by sampling random values in local values

      determine cutoff based on sample and add to list of cutoffs

      send determined cutoffs to other processess
      receive cutoffs from other processess

      sequentially sort cuttoffs

      create lists of values by comparing to cutoffs and add to send buffers
      send buffers to all other processess
      receive buffers from all other processess and append to local values
      
      sequentially sort local values

### Bitonic Sort:
    bitonic_sort_step(device values, j, k)
      i = index of the current thread + number of threads per block + the index of the current block
      ixj = i^j

      if ixj > i
        if i&k==0
          if device values at i > device values at ixj
            swap(i,ixj)

      if i&k!=0
        if device values at i < device values at ixj
          swap(i,ixj)

    bitonic_sort(values, device values, size, threads)
      copy values from host to device as device values

      blocks = size/threads
      for k = 2 to size, double k each iteration
        for j = half k to 0, halve k each iteration
          bitonic_sort_step(device values, j, k) call to device code

      copy device values from device to host as values

## 3. _due 11/08_ Pseudocode for each algorithm and implementation

### Sample Sort (MPI):
    sample_sort(NUM_VALS, list local_values, local_size, num_procs, rank, sample_size)
      for i = 0 to local_size
        populate local_size[i]

      sample = list of floats with size of sample_size

      if not rank 0
        for i = 0 to sample_size
          int sample_index = random index within local_size
          sample[i] = local_values at sample_index

          sequentially sort sample

          cutoff largest values in sample
      else
          cutoff = 0

      cutoffs = list of floats
      for i = 0 to num_procs
        if i is not rank
          send cutoff to rank i

          recv_cutoff
          recieve recv_cutoff from rank i
          append recv_cutoff to cutoffs
        else
          append cutoff to cutoffs

      sequentially sort cuttoffs

      list of (float, int) pairs cutoff_pairs
      for i = 0 to num_procs
        cutoff_pair = (cutoffs[i], i)
        append cutoff_pair to cutoff_pairs

      for i = 0 to num_procs
        send_buf = list of flaots
        append send_buf to send_bufs

        receive_buf = list of flaots
        append receive_buf to receive_bufs

      list of floats local_buf
      for i = 0 to local_size
        for j = num_procs - 1 to 0
          if local_values[i] >= cutoff_pairs[j].first
            if j != rank
              push local_values at i to send_bufs[j]
            else
              append local_values[i] to local_buf
            break

      clear local_values
      append all of local_buf to local_values

      for i = 0 to num_procs
        if i != rank
          send size of send_buf[i] to rank i
          receive size of recv_buf[i] from rank i

          send send_buf[i] to rank i
          receive recv_buf[i] from rank i into local_values

      sequentially sort local_values

### Bitonic Sort (CUDA):
    bitonic_sort_step(dev_values, j, k)
      i = threadIdx.x + blockDim.x * blockIdx.x
      ixj = i^j

      if ixj>i
        if i&k==0
          if dev_values[i]>dev_values[ixj]
            exchange(i,ixj)

      if i&k!=0
        if dev_values[i]<dev_values[ixj]
          exchange(i,ixj)

    bitonic_sort(values, dev_values, NUM_VALS, threads)
      copy values from host to device

      blocks = NUM_VALS/threads
      for k = 2 to NUM_VALS, k <<= 1
        for j=k>>1 to 0, j=j>>1
          bitonic_sort_step(dev_values, j, k) call to device code

      copy dev_values from device to host

### Mergesort (MPI)
    mergesort() (pseudocode in section 2)
        
    MPImergesort(arr, num_vals, num_procs)
        generate a binary tree with a leaf for every process, so that the root has value zero and for every parent, one child has the same value as the parent and one child has a new value
        node thisProcsLeaf = searchForLeaf(rank)
        destList = walkUpTree(thisProcsLeaf) 
        set work = 1
        set local_size = num_vals / num_procs
        while work == 1:
            mergesort(local_values)
            nextDest = destList.pop()
            if(nextDest == rank)
                receive a message from another processor sending this process data
                combined_vals = her_vals + local_vals
                mergesort(combined_vals, 0, local_size * 2)
                local_vals = combined_vals
            else
                send our sorted local data to the processor with rank == nextDest
                work = 0 (this process does no more computation)

            if(destList.empty())
                work = 0 (the root process has finished sorting the array)



### Mergesort (CUDA)
    mergesort() (pseudocode in section 2)
    
    CUDAmergesortStep(arr, num_vals, sectionWidth)
        set start = sectionWidth * threadID
        set end = left + sectionWidth
        mergesort(arr, start, end)

    CUDAmergesort(arr, num_vals, num_threads, num_blocks)
        set sliceWidth = 2
        set threadsToUse = num_threads
        if(threadsToUse > num_vals / sliceWidth)
            threadsToUse = num_vals / sliceWidth
        if(threadsToUse < num_vals / sliceWidth)
            sliceWidth = num_vals / threadsToUse
        forever
            CUDAMergesortStep<<<num_blocks, threadsToUse>>>(arr, num_vals, sliceWidth)
            if(threadsToUse == 1)
                break
            sliceWidth *= 2
            threadsToUse /= 2

### Odd even sort (CUDA)
```
  function oddeven(data):
    for i in [0,N):
      oddeven_dev<<<blocks, threads>>>(data, i%2);

  function oddeven_dev(data, phase):
    idx = threadIdx + blockDim * blockIdx;
    if (phase==0 and idx%2==0) or (phase==1 and idx%2==1):
      if data[idx+1] < data[idx]:
        swap data at idx+1,idx
```
### Odd even sort (MPI)
```
  function rightSide(local, local_size, rank):
    Send local[rank][0] to rank-1
    receive new local[rank][0] from rank-1

  function leftSide(local, local_size, rank):
    Receive local[rank+1][0] from rank+1
    sort local[rank][-1], local[rank+1][0]
    send new local[rank+1][0] to rank+1

  function oddeven(local_data, local_size, nproc, rank):
    for i in [0,N):
      for j in [i%2,local_size):
        sort local_data[j], local_data[j+1]
      MPI_BARRIER
      if local_size%2==1:
        if i%2 == 0:
          if rank%2==0:
            rightSide(local_data, local_size, rank)
          else:
            leftSide(local_data, local_size, rank)
        else:
          if rank%2==0:
            leftSide(local_data, local_size, rank)
          else:
            rightSide(local_data, local_size, rank)
      else:
        if i%2==1:
          if rank%2==0:
            leftSide(local_data, local_size, rank)
          else:
            rightSide(local_data, local_size, rank)
      
      MPI_BARRIER
```

## 3. _due 11/08_ Evaluation plan - what and how will you measure and compare
We will evaluate the performance in terms of variable problem size and number of threads for the CUDA algorithms. We will evaluate the performance by Weak Scaling for the MPI algorithms.

