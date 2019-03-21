#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
//#include <cub/cub.cuh>
//#include "Utilities.cuh"

using namespace std;

#define CONNECTIONS "data.csv"
#define BLOCKSIZE 256
#define PRECISION 1000000
#define THRESHOLD 0.000001

//const int N = 256;

// Default reduction kernel from seminar slides - check for possible optimizations
template <unsigned int blockSize> __global__ void cuda_reduction(double *array_in, double *reduct, size_t array_len) {
    extern volatile __shared__ double sdata[];
    
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

template <unsigned int blockSize> __global__ void damping_reduction(double *array_in, double *reduct, double *factor, size_t array_len) {
	extern volatile __shared__ double sdata[];
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
template <unsigned int blockSize> __global__ void weighted_sum_partial(double *pagerank_in, double *reduct,
	int *column, double *mat_data, size_t row_len, size_t pk_len){
	extern volatile __shared__ double sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while (i < row_len) {
		if (column[i] >= pk_len) printf("%d\n", column[i]);
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

template <unsigned int blockSize> __global__ void handle_multiply(double *old_pk, double *new_pk, double *damp,
	int *row_indices, int *columns, double *mat_data, size_t pk_len){	
		
	int tid = blockIdx.x*blockSize +threadIdx.x;	
	if (tid < pk_len){

		// Sum "damping" contribution
		new_pk[tid] = old_pk[tid] + *damp;
		//printf("tid: %d", tid);

		int row_len = row_indices[tid+1] - row_indices[tid];
		// If there is data for the row...
		if (row_len > 0){
			double *mult, *result;
			int block_number = (row_len + BLOCKSIZE - 1) / BLOCKSIZE;
			cudaMalloc(&mult, sizeof(double)*block_number);
			cudaMalloc(&result, sizeof(double)); //error??????????????

			int index = row_indices[tid]; // Index of the first element of the row in columns array and data array
			
			weighted_sum_partial <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(old_pk, mult, &columns[index], &mat_data[index], row_len, pk_len);
			cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(mult, result, block_number);
			cudaDeviceSynchronize();
			
			new_pk[tid] += *result;
			cudaFree(mult);
			cudaFree(result);
			
		}
	}
}


template <unsigned int blockSize> __global__ void check_termination(double *old_pk, double *new_pk, bool *loop){
	int index = blockSize * gridDim.x + threadIdx.x;
	if (fabs(floor( (new_pk[index] - old_pk[index]) * PRECISION ) / PRECISION) > THRESHOLD){
				*loop = true;
	}
}

template <unsigned int blockSize> __global__ void sauron_eye(double *old_pk, double *new_pk, int *row_indices, int *columns,
	double *data, double *damping, int *pk_len, int *data_len){

	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;
	printf("Block number: %d\n", block_number);
	
	double *damp_res, *out;
	bool *loop;
	cudaMalloc(&damp_res, sizeof(double));
	cudaMalloc(&loop, sizeof(bool));
	cudaMalloc(&out, sizeof(double)*block_number);
	
	int i = 0;

	*loop = true;

	while (*loop){
		printf("Iteration %d\n", i);
		i++;

		printf("len :%d\n", *pk_len);
		// Calculate "damping contribution"
		printf("Begin damping\n");
		cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(old_pk, out, *pk_len);
		damping_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>> (out, damp_res, damping, block_number);
		cudaDeviceSynchronize();

		printf("Begin multiply\n");
		handle_multiply<BLOCKSIZE> <<<block_number, BLOCKSIZE>>> (old_pk, new_pk, damp_res, row_indices, columns, data, *pk_len);
		cudaDeviceSynchronize();
		
		*loop = false;
		printf("Begin check\n");
		check_termination<BLOCKSIZE> <<<block_number, BLOCKSIZE>>>(old_pk, new_pk, loop);
		cudaDeviceSynchronize();
	}

	cudaFree(damp_res);
	cudaFree(loop);

}

int main(){

   	ifstream connFile, probFile;
   	connFile.open(CONNECTIONS);
	int *row_ptrs, *col_indices, *connections;
	int nodes_number, col_indices_number, conn_size, row_len;
	double damping;

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
		connections = (int *) malloc(conn_size*sizeof(int));

		// Store data
		getline(connFile, line);
		stringstream uu(line);
		for (int i = 0; i < conn_size; i++){
			getline(uu, element, ',');
			connections[i] = stoi(element);
		}

		// Save "damping" matrix factor
		getline(connFile, line);
		damping = stod(line);
		connFile.close();
	}

	double pr[nodes_number];
	double uniform_p = 1/(double)nodes_number;
	for (int i = 0; i < nodes_number; i++){
		pr[i] = uniform_p;
	}

	double *pk_gpu, *new_pk, *factor_gpu, *d_gpu;
	int *c_gpu, *r_gpu, *data_len, *pk_len;

	// Allocate device memory

	cudaMallocManaged(&pk_gpu, nodes_number*sizeof(double));
	cudaMallocManaged(&new_pk, sizeof(double)*nodes_number);
	cudaMallocManaged(&factor_gpu, sizeof(double));
	cudaMallocManaged(&c_gpu, sizeof(int)*col_indices_number);
	cudaMallocManaged(&d_gpu, sizeof(double)*col_indices_number);
	cudaMallocManaged(&r_gpu, sizeof(int)*(nodes_number+1));
	cudaMallocManaged(&pk_len, sizeof(int));
	cudaMallocManaged(&data_len, sizeof(int));

	// Populate device data from main memory

	cout << nodes_number << endl;

	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(factor_gpu, &damping, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu, col_indices, sizeof(int)*nodes_number, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gpu, connections, sizeof(double)*nodes_number, cudaMemcpyHostToDevice);
	cudaMemcpy(r_gpu, row_ptrs, sizeof(int)*(nodes_number+1), cudaMemcpyHostToDevice);
	cudaMemcpy(pk_len, &nodes_number, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data_len, &conn_size, sizeof(int), cudaMemcpyHostToDevice);

	// Calculate constant weighted pagerank sum
	
	//cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, nodes_number);
	//cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(out, pk_gpu, block_number);

	//cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, nodes_number);
	//damping_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>> (out, pk_gpu, factor_gpu, block_number);


	// weighted_sum_partial <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, c_gpu, d_gpu, 5, nodes_number);
	// cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(out, pk_gpu, block_number);
	// handle_multiply<BLOCKSIZE> <<<block_number, BLOCKSIZE>>> (pk_gpu, out, uniform_gpu, r_gpu, c_gpu, d_gpu, nodes_number);
	// cudaDeviceSynchronize();

	sauron_eye<1><<<1,1>>>(pk_gpu, new_pk, r_gpu, c_gpu, d_gpu, factor_gpu, pk_len, data_len);

	cudaDeviceSynchronize();
	cudaMemcpy(pr, new_pk, nodes_number*sizeof(double), cudaMemcpyDeviceToHost);

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