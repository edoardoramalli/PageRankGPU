#include <stdio.h>
#include <iostream>
using namespace std;

template <unsigned int blockSize> __device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize> __global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; 
    sdata[tid] = 0;

    while (i < n) { 
        sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
        i += gridSize; 
    } 

    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); } 
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); } 
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32){
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    } //warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(){

	float *array_in, *array_out;
	size_t array_length = 100000;
	const unsigned int grid_Size = 256, block_Size = 512;
	cudaMallocManaged(&array_in, array_length * sizeof(float));
	cudaMallocManaged(&array_out, array_length * sizeof(float));

	float cpu_sum = 0.0f;
	for(size_t i = 0; i < array_length; i++) {array_in[i] = 1; cpu_sum += 1;}
	// Calls the reduction kernel
	reduce6 < block_Size > 
		<<< grid_Size, block_Size, block_Size * sizeof(float) >>> 
		(array_in,array_out, array_length);
	reduce6 < block_Size > 
		<<< 1, block_Size, block_Size * sizeof(float) >>> 
		(array_out, array_in,grid_Size);
	cudaDeviceSynchronize();
	// Prints grid_Size elements of array_out
	std::cout << "cpu: " << cpu_sum << std::endl;
	std::cout << "gpu: " << array_in[0] << std::endl;
	cudaFree(array_in);
	cudaFree(array_out);
    return 0;
    
}