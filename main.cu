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
    if (tid == 0) reduct[blockIdx.x] = sdata[0];
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
	int *column, double *mat_data, size_t rows_number, size_t pk_len){
	extern volatile __shared__ double sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while (i < rows_number) {
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
		// Need to weight sum this too!
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

		int row_len = row_indices[tid+1] - row_indices[tid];

		// If there is data for the row...
		if (row_len != 0){
			double *mult, result;
			cudaMalloc(&mult, sizeof(double)*row_len);

			int index = row_indices[tid]; // Index of the first element of the row in columns array and data array
			
			int block_number = (row_len + BLOCKSIZE - 1) / BLOCKSIZE;
			
			weighted_sum_partial <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(old_pk, mult, &columns[index], &mat_data[index], row_len, pk_len);
			cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(mult, &result, block_number);
			cudaDeviceSynchronize();
			
			new_pk[tid] += result;
		}
	}
}


template <unsigned int blockSize> __global__ void check_termination(double *old_pk, double *new_pk){
	extern volatile __shared__ bool terminate;
	terminate = true;
	int index = blockSize * gridDim.x + threadIdx.x;
	if (fabs(floor( (new_pk[index] - old_pk[index]) * PRECISION ) / PRECISION) > THRESHOLD){
				terminate = false;

	}
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

		// Store column indices
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
	//nodes_number = 1024;

	double pr[nodes_number];
	double uniform_p = 1/(double)nodes_number;
	double cpu_sum = 0;
	for (int i = 0; i < nodes_number; i++){
		pr[i] = uniform_p;
		//cpu_sum += uniform_p;
	}

	int block_number = (nodes_number + BLOCKSIZE - 1) / BLOCKSIZE;

	double *pk_gpu, *out, *uniform_gpu, *new_pk, *factor_gpu, *d_gpu;
	int *w_gpu;
	//cout << cpu_sum << endl;

	cudaMallocManaged(&pk_gpu, nodes_number*sizeof(double));
	cudaMallocManaged(&out, nodes_number*sizeof(double));
	cudaMallocManaged(&uniform_gpu, sizeof(double));
	cudaMallocManaged(&new_pk, sizeof(double)*nodes_number);
	cudaMallocManaged(&factor_gpu, sizeof(double));
	cudaMallocManaged(&w_gpu, sizeof(int)*10);
	cudaMallocManaged(&d_gpu, sizeof(double)*10);

	//for (int i = 0; i < nodes_number; i++) pk_gpu[i] = uniform_p;
	nodes_number = 10;
	int w[nodes_number];
	double d[nodes_number];
	for (int i = 0; i < nodes_number; i++){
		pr[i] = i;
		if (i % 2 == 0){
			w[i/2] = i;
			d[i/2] = i;
			cpu_sum += i*i;
		}
	} 

	block_number = (nodes_number + BLOCKSIZE - 1) / BLOCKSIZE;

	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(uniform_gpu, uniform_p, sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(factor_gpu, &damping, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu, w, sizeof(int)*10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gpu, d, sizeof(double)*10, cudaMemcpyHostToDevice);

	// Calculate constant weighted pagerank sum
	
	//cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, nodes_number);
	//cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(out, pk_gpu, block_number);

	//cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, nodes_number);
	//damping_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>> (out, pk_gpu, factor_gpu, block_number);


	weighted_sum_partial <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, w_gpu, d_gpu, 5, nodes_number);
	cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(out, pk_gpu, block_number);
	cudaDeviceSynchronize();
	cudaMemcpy(pr, pk_gpu, nodes_number*sizeof(double), cudaMemcpyDeviceToHost);

	cout << pr[0] << endl;
	cout << cpu_sum << endl;




	//cudaFree(pk);
	cudaFree(uniform_gpu);
	cudaFree(new_pk);
	cudaFree(pk_gpu);
	cudaFree(out);
	
	return 0;
}