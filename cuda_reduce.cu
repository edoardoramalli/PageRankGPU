
#include <stdio.h>

#define BLOCKSIZE 32
#define PRECISION 1000000
#define THRESHOLD 0.000001
#define DAMPING_F 0.85


template <unsigned int blockSize> __device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__device__ inline void ouratomicAdd(float* address, float value){

  float old = value;  
  float new_old;

  do{
	new_old = atomicExch(address, 0.0f);
	new_old += old;
  }
  while ((old = atomicExch(address, new_old))!=0.0f);
}

template <unsigned int blockSize> __global__ void pk_multiply(float* data, int* columns, int* rows, float* old_pk, float* new_pk, unsigned int len, int *pk_len){
	size_t tid = threadIdx.x,    
	i = blockIdx.x * blockSize + tid;
	
	if(i < len){
		float sum = data[i] * old_pk[columns[i]];
		//new_pk[rows[i]] += sum;
		ouratomicAdd(&new_pk[rows[i]], sum);
		//if (rows[i] == 5) printf("cacca %.10f\n", sum);
		
	}
}

template <unsigned int blockSize> __global__ void sumAll(float *empty_contrib, float *damping_matrix, float *new_pk, int *pk_len){
	size_t tid = threadIdx.x,    
	i = blockIdx.x * blockSize + tid;
	
	if(i < *pk_len){
		new_pk[i] += *empty_contrib + *damping_matrix;
	}
}

template <unsigned int blockSize> __global__ void cuda_reduction(float *array_in, float *reduct, size_t array_len) {
	/*Parallel block reduction*/

    extern volatile __shared__ float sdata[];
    
    size_t tid = threadIdx.x,
    gridSize = blockSize * gridDim.x,
    
    i = blockIdx.x * blockSize + tid;
    
    sdata[tid] = 0;
    
    while (i < array_len) {
        sdata[tid] += array_in[i];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) {
		warpReduce<blockSize>(sdata, tid);
	}
	
    if (tid == 0){
		reduct[blockIdx.x] = sdata[0];
	} 
}

template <unsigned int blockSize> __global__ void uniform_reduction(float *old_pk, int *empty_cols,float *reduct, float *factor, size_t array_len) {
	/* Calculate contribution from empty columns to each line:
	sum pagerank at index equal to empty column index in T',
	then multiply by the teleportation probability */

    extern volatile __shared__ float sdata[];
    
    size_t tid = threadIdx.x,
    gridSize = blockSize * gridDim.x,
    
    i = blockIdx.x * blockSize + tid;
    
	sdata[tid] = 0;
    
    while (i < array_len) {
        sdata[tid] += old_pk[empty_cols[i]];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) {
		warpReduce<blockSize>(sdata, tid);
	}
	
    if (tid == 0){
		reduct[blockIdx.x] = (*factor)*sdata[0];
	} 
}

template <unsigned int blockSize> __global__ void weighted_sum_partial(float *pagerank_in, float *reduct,
	int *column, float *mat_data, size_t row_len, size_t pk_len){
	/* Perform first step of pagerank row by column product
	by multiplying T' row by pagerank elements, only for non null T' elements.
	T is input as a CSR matrix (3 arrays: row pointers, columns, data)
	*/

	extern volatile __shared__ float sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize *gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while (i < row_len) {
		sdata[tid] += mat_data[i]*pagerank_in[column[i]];
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid <  64) sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	if (tid < 32) {
		warpReduce<blockSize>(sdata, tid);
	}
	if (tid == 0) reduct[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize> __global__ void termination_reduction(float *new_pk, float *old_pk, float *reduct, size_t array_len) {
    extern volatile __shared__ float sdata[];
    
    size_t tid = threadIdx.x,
    gridSize = blockSize * gridDim.x,
    
    i = blockIdx.x * blockSize + tid;
    
    sdata[tid] = 0;
    
    while (i < array_len) {
		float diff = new_pk[i] - old_pk[i];
        sdata[tid] += diff*diff;
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) {
		warpReduce<blockSize>(sdata, tid);
	}
	
    if (tid == 0){
		reduct[blockIdx.x] = sdata[0];
	} 
}


template <unsigned int blockSize> __global__ void check_termination(float *old_pk, float *new_pk, float* out, float* result, bool *loop, 
	int *pk_len, size_t out_len){
	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;

	termination_reduction <BLOCKSIZE> <<<block_number, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>> (new_pk, old_pk, out, *pk_len);
	cuda_reduction <BLOCKSIZE> <<<1, BLOCKSIZE, BLOCKSIZE*sizeof(float) >>> (out, result, out_len);
	cudaDeviceSynchronize();
	float error;
	error = sqrtf(*result);
	printf("Error  %.10f\n", error);
	if (error > THRESHOLD) {
		*loop = true;
	}

}