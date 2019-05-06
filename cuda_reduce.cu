
#include <stdio.h>

#define BLOCKSIZE 128

/* Functions involving vector reduction are replicated in order to avoid
 control flow alterations that would cause warp inefficiency*/

template <unsigned int blockSize> __device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    /* Loop unrolling performed in device code
    to optimize reduction performance */

    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// __device__ inline void floatAtomicAdd(float* address, float value){

//     /*  Try writing on atomic variable until writing is truly performed */

//     float old = value;  
//     float new_old;

//     do{
//         new_old = atomicExch(address, 0.0f);
//         new_old += old;
//     }
//     while ((old = atomicExch(address, new_old))!=0.0f);
// }

template <unsigned int blockSize> __global__ void pkMultiply(float * __restrict__ data, int * __restrict__ columns, int * __restrict__ rows, float * __restrict__ old_pk, float *new_pk, unsigned int len, int * __restrict__ pk_len){
    /*
    Taking as input data, columns and rows from a (row, column) --> data representation
    perform partial row by column matrix multiplication element by element, 
    where row elements are not null.
    Add multiplication result to new pagerank vector at row equal to the
    row of the matrix we are currently considering
    */
    
    size_t tid = threadIdx.x,    
	i = blockIdx.x * blockSize + tid;
	
	if(i < len){
		float sum = data[i] * old_pk[columns[i]];
		atomicAdd(&new_pk[rows[i]], sum);		
	}
}

template <unsigned int blockSize> __global__ void sumAll(float * __restrict__ empty_contrib, float * __restrict__ damping_matrix, float *new_pk, int * __restrict__ pk_len){
    /* Sum all the partial contributions:
        - empty columns contribute, in CSR representation of T transposed 
        (before adding teleportation probabilities) are discarded. These columns generate a 
        contribute equal for all rows and constant within each iteration.
        - damping matrix: since the sum of all elements of PageRank vector equals 1,
        the product with a constant (throughout all iterations) matrix filled with equal values 
        is a vector filled with the same value for all iterations. This value is calculated once in 
        preprocessing steps.
        - new pagerank: values previously calculated by pk_multiply
    */
    
    size_t tid = threadIdx.x,    
	i = blockIdx.x * blockSize + tid;
	
	if(i < *pk_len){
		new_pk[i] += *empty_contrib + *damping_matrix;
	}
}

template <unsigned int blockSize> __global__ void cudaReduction(float *array_in, float *reduct, size_t __restrict__ array_len) {
    /*Parallel block reduction from 
    CUDA seminar tutorial (kernel #7)
    https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    */

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

template <unsigned int blockSize> __global__ void uniformReduction(float * __restrict__ old_pk, int * __restrict__ empty_cols, float *reduct, float * __restrict__ factor, size_t __restrict__ array_len) {
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

template <unsigned int blockSize> __global__ void terminationReduction(float * __restrict__ new_pk, float * __restrict__ old_pk, float * __restrict__ reduct, size_t __restrict__ array_len) {
    extern volatile __shared__ float sdata[];

    /* Calculate reduction to block size performing 
    difference of two source vectors */
    
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

template <unsigned int blockSize> __global__ void checkTermination(float * __restrict__ old_pk, float * __restrict__ new_pk, float * out, float* result, bool *loop, 
    
    /* Check L2 norm of difference of pageRank vectors,
    loop variable is set true only if precision has not been reached */
    
    int *pk_len, size_t out_len, float *precision){
	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;

	terminationReduction <BLOCKSIZE> <<<block_number, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>> (new_pk, old_pk, out, *pk_len);
	cudaReduction <BLOCKSIZE> <<<1, BLOCKSIZE, BLOCKSIZE*sizeof(float) >>> (out, result, out_len);
	cudaDeviceSynchronize();
	
	float error = sqrtf(*result);
	if (error >= *precision) {
		*loop = true;
	}

}