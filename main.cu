#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
//#include <cub/cub.cuh>
//#include "Utilities.cuh"

using namespace std;

#define CONNECTIONS "data.csv"
#define BLOCKSIZE 8
#define PRECISION 1000000
#define THRESHOLD 0.000001

//const int N = 256;

// Default reduction kernel from seminar slides - check for possible optimizations

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <unsigned int blockSize> __global__ void cuda_reduction(float *array_in, float *reduct, size_t array_len) {
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
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	
    if (tid == 0){
		//printf("%d, %d\n", blockIdx.x, array_len);
		reduct[blockIdx.x] = sdata[0];
	} 
}

template <unsigned int blockSize> __global__ void damping_reduction(float *array_in, float *reduct, float *factor, size_t array_len) {
	extern volatile __shared__ float sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
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
		if (tid <  64) sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) reduct[blockIdx.x] = sdata[0]*(*factor);
}

// Perform first step of pagerank row by column product
template <unsigned int blockSize> __global__ void weighted_sum_partial(float *pagerank_in, float *reduct,
	int *column, float *mat_data, size_t row_len, size_t pk_len){
	extern volatile __shared__ float sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
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
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) reduct[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize> __global__ void handle_multiply(float *old_pk, float *new_pk, float *damp,
	int *row_indices, int *columns, float *mat_data, size_t pk_len){	
		
	int tid = blockIdx.x*blockSize +threadIdx.x;	
	if (tid < pk_len){

		// Sum "damping" contribution
		new_pk[tid] = old_pk[tid] + *damp;
		//printf("tid: %d", tid);
		int index = row_indices[tid];  // Index of the first element of the row in columns array and data array
		int row_len = row_indices[tid+1] - index;
		// If there is data for the row...
		if (row_len > 0){
			float *mult, *result;
			int block_number = (row_len + BLOCKSIZE - 1) / BLOCKSIZE;
			cudaMalloc(&mult, sizeof(float)*block_number);
			cudaMalloc(&result, sizeof(float));
			
			//printf("threads %d\n", row_len);

			weighted_sum_partial <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>>(old_pk, mult, &columns[index], &mat_data[index], row_len, pk_len);
			__syncthreads();
			//printf("thread %d completed weighted sum!\n", tid);
			cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>>(mult, result, block_number);
			__syncthreads();
			
			new_pk[tid] += *result;
			cudaFree(mult);
			cudaFree(result);
			
		}
	}
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
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	
    if (tid == 0){
		//printf("%d, %d\n", blockIdx.x, array_len);
		reduct[blockIdx.x] = sdata[0];
	} 
}


template <unsigned int blockSize> __global__ void check_termination(float *old_pk, float *new_pk, float* out, float* result, bool *loop, 
	int *pk_len, size_t out_len){
	// int index = blockSize * gridDim.x + threadIdx.x;
	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;

	termination_reduction <BLOCKSIZE> <<<block_number, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (new_pk, old_pk, out, *pk_len);
	__syncthreads();
	cuda_reduction <BLOCKSIZE> <<<1, BLOCKSIZE, BLOCKSIZE*sizeof(float) >>> (out, result, out_len);
	
	float error;
	error = sqrtf(*result);
	if (error > THRESHOLD) {
		printf("Error  %f\n", error);
		*loop = true;
	}
	
	// if (fabs(floor( (new_pk[index] - old_pk[index]) * PRECISION ) / PRECISION) > THRESHOLD){
	// 			*loop = true;
	// }
}
//sqrt(sum((y-x)**2))

template <unsigned int blockSize> __global__ void sauron_eye(float *old_pk, float *new_pk, int *row_indices, int *columns,
	float *data, float *damping, int *pk_len, int *data_len){

	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;
	//printf("Block number: %d\n", block_number);
	
	float *result, *out;
	bool *loop;
	cudaMalloc(&result, sizeof(float));
	cudaMalloc(&loop, sizeof(bool));
	cudaMalloc(&out, sizeof(float)*block_number);
	
	int i = 0;

	float * tmp;


	*loop = true;

	while (*loop){
		printf("Iteration %d\n", i);
		if (i%2!=0){
			tmp = old_pk;
			old_pk = new_pk;
			new_pk = tmp;
		}

		//printf("len :%d\n", *pk_len);
		// Calculate "damping contribution"
		printf("Begin damping\n");
		cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>>(old_pk, out, *pk_len);
		damping_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (out, result, damping, block_number);
		cudaDeviceSynchronize();

		printf("Begin multiply\n");
		handle_multiply<BLOCKSIZE> <<<block_number, BLOCKSIZE>>> (old_pk, new_pk, result, row_indices, columns, data, *pk_len);
		cudaDeviceSynchronize();
		
		*loop = false;
		printf("Begin check\n");
		check_termination<1> <<<1, 1>>>(old_pk, new_pk, out, result, loop, pk_len, block_number);

		// template <unsigned int blockSize> __global__ void check_termination(float *old_pk, float *new_pk, float* out, float* result, bool *loop, 
		// 	size_t pk_len, size_t out_len){
		cudaDeviceSynchronize();
		i++;
	}

	cudaFree(result);
	cudaFree(loop);
	cudaFree(out);

}

int main(){

   	ifstream connFile, probFile;
   	connFile.open(CONNECTIONS);
	int *row_ptrs, *col_indices;
	float *connections;
	int nodes_number, col_indices_number, conn_size, row_len;
	float damping;

	cout << "Load connections" << endl;;
	if (connFile){
		string line, element;
		
		// Read row_len number and allocate vector
		getline(connFile, line);
		row_len = stoi(line);
		nodes_number = row_len-1;
		row_ptrs = (int *) malloc(row_len*sizeof(int));
		
		// Store meaningful rows
		getline(connFile, line);
		stringstream ss(line);
		for (int i = 0; i < row_len; i++){
			getline(ss, element, ',');
			row_ptrs[i] = stoi(element);
		}

		// Read column indices number and allocate vector
		getline(connFile, line);
		col_indices_number = stoi(line);
		col_indices = (int *) malloc(col_indices_number*sizeof(int));

		// Store column indices
		getline(connFile, line);
		stringstream tt(line);
		for (int i = 0; i < col_indices_number; i++){
			getline(tt, element, ',');
			col_indices[i] = stoi(element);
		}

		// Read data length
		getline(connFile, line);
		conn_size = stoi(line);
		cout << "Conn size: " << conn_size << ", column_size:  " << col_indices_number << endl;
		connections = (float *) malloc(conn_size*sizeof(float));

		// Store data
		getline(connFile, line);
		stringstream uu(line);
		for (int i = 0; i < conn_size; i++){
			getline(uu, element, ',');
			connections[i] = stod(element);
		}

		// Save "damping" matrix factor
		getline(connFile, line);
		damping = stod(line);
		connFile.close();
	}

	float pr[nodes_number];
	float uniform_p = 1/(float)nodes_number;
	for (int i = 0; i < nodes_number; i++){
		pr[i] = uniform_p;
	}

	float *pk_gpu, *new_pk, *factor_gpu, *d_gpu;
	int *c_gpu, *r_gpu, *data_len, *pk_len;

	// Allocate device memory

	cudaMalloc(&pk_gpu, nodes_number*sizeof(float));
	cudaMalloc(&new_pk, sizeof(float)*nodes_number);
	cudaMalloc(&factor_gpu, sizeof(float));
	cudaMalloc(&c_gpu, sizeof(int)*col_indices_number);
	cudaMalloc(&d_gpu, sizeof(float)*col_indices_number);
	cudaMalloc(&r_gpu, sizeof(int)*(nodes_number+1));
	cudaMalloc(&pk_len, sizeof(int));
	cudaMalloc(&data_len, sizeof(int));

	// Populate device data from main memory

	cout << nodes_number << endl;


	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(float), cudaMemcpyHostToDevice);
	//cout << "1" << endl;
	cudaMemcpy(factor_gpu, &damping, sizeof(float), cudaMemcpyHostToDevice);
	//cout << "2" << endl;
	cudaMemcpy(c_gpu, col_indices, sizeof(int)*col_indices_number, cudaMemcpyHostToDevice);
	// cout << "3" << endl;
	cudaMemcpy(d_gpu, connections, sizeof(float)*col_indices_number, cudaMemcpyHostToDevice);
	// cout << "4" << endl;
	cudaMemcpy(r_gpu, row_ptrs, sizeof(int)*(nodes_number+1), cudaMemcpyHostToDevice);
	// cout << "5" << endl;
	cudaMemcpy(pk_len, &nodes_number, sizeof(int), cudaMemcpyHostToDevice);
	// cout << "6" << endl;
	cudaMemcpy(data_len, &conn_size, sizeof(int), cudaMemcpyHostToDevice);
	// cout << "7" << endl;


	sauron_eye<1><<<1,1>>>(pk_gpu, new_pk, r_gpu, c_gpu, d_gpu, factor_gpu, pk_len, data_len);

	gpuErrchk( cudaDeviceSynchronize() );

	// Copy data back
	cudaMemcpy(pr, new_pk, nodes_number*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(new_pk);
	cudaFree(pk_gpu);
	cudaFree(r_gpu);
	cudaFree(c_gpu);
	cudaFree(d_gpu);
	cudaFree(factor_gpu);
	cudaFree(pk_len);
	cudaFree(data_len);
	
	return 0;
}