# CSCE 435 Group project

## 0. Group number: 

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
- Mergesort (CUDA, MPI)

## 2. Pseudocode

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
          if dev_values[i] dev_values[ixj]
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
          
## 3. _due 11/08_ Pseudocode for each algorithm and implementation
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
        

## 3. _due 11/08_ Evaluation plan - what and how will you measure and compare

For example:
- Effective use of a GPU (play with problem size and number of threads)
- Strong scaling to more nodes (same problem size, increase number of processors)
- Weak scaling (increase problem size, increase number of processors)
