#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
//#include <cub/cub.cuh>
//#include "Utilities.cuh"

using namespace std;

#define CONNECTIONS "data.csv"
#define BLOCKSIZE 256

//const int N = 256;

// Unoptimized kernel
__global__ void addReduceKernel(double *inData, double *outData){
	extern __shared__ double sharedData[];

	// Each thread copies one element to shared memory from global memory
	unsigned int tid = threadIdx.x;
	unsigned int i= blockIdx.x*blockDim.x+ threadIdx.x;
	sharedData[tid] = inData[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2){
		if(tid % (2*s) == 0){
			sharedData[tid] += sharedData[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if(tid == 0) outData[blockIdx.x] = sharedData[0];
}

// Default reduction kernel from seminar slides - check for possible optimizations
template <unsigned int blockSize> __global__ void cuda_reduction(double *array_in, double *reduct, size_t array_len) {
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
	if (tid == 0) reduct[blockIdx.x] = sdata[0];
}


// Perform first step of pagerank row by column product
template <unsigned int blockSize> __global__ void weighted_sum_partial(double *reduct, double *pagerank, 
	double *column, double *mat_data, size_t array_len, size_t pk_len){
	extern volatile __shared__ double sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while (i < array_len) {
		sdata[tid] += mat_data[i]*pagerank[column[i]];
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

template <unsigned int blockSize> __global__ void handle_multiply(double *reduct, double *pagerank,
	double *row_indices, double *columns, double *mat_data,
	size_t indices_len, size_t len_data, size_t array_len, size_t pk_len){
		for (i = 0; i < indices_len-1; i++){
			// Probably adding divergence, would it be meaningful to perform it directly into GPU?
			if (row_indices[i] == row_indices[i+1]){
				// Uniform reduction
			}
			else{
				// "True" pagerank product
				int index = row_indices[i];
				int data_number = row_indices[i+1] - index;
				// We need to allocate space for reduction, also how to select blocksize = data_number?
				<BLOCKSIZE><data_number)BLOCKSIZE),BLOCKSIZE, BLOCKSIZE*sizeof(double)>weighted_sum_partial(reduct, pagerank
				columns[index], mat_data[index],data_number);
			}
		}

}
int main(){

   	ifstream connFile, probFile;
   	connFile.open(CONNECTIONS);
	int *row_ptrs, *col_indices, *connections;
	int nodes_number, col_indices_number, conn_size, row_len;

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
		connFile.close();
	}

	double pr[nodes_number];
	double uniform_p = 1/(double)nodes_number;
	double cpu_sum = 0;
	for (int i = 0; i < nodes_number; i++){
		pr[i] = uniform_p;
		//cpu_sum += uniform_p;
	}

	double *pk_gpu, *out, *uniform_gpu, *new_pk;
	//cout << cpu_sum << endl;

	cudaMallocManaged(&pk_gpu, nodes_number*sizeof(double));
	cudaMallocManaged(&out, nodes_number*sizeof(double));
	cudaMallocManaged(&uniform_gpu, sizeof(double));
	cudaMallocShared(&new_pk, sizeof(double)*nodes_number)

	for (int i = 0; i < nodes_number; i++) pk_gpu[i] = uniform_p;
	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(uniform_gpu, uniform_p, sizeof(double), cudaMemcpyHostToDevice);

	// Calculate constant weighted pagerank sum
	
	cuda_reduction <BLOCKSIZE> <<< nodes_number/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, nodes_number);
	cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(out, pk_gpu, nodes_number);
	//addReduceKernel<<<nodes_number/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(pk_gpu, out);
	//addReduceKernel<<<1, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(out, pk_gpu);
	cudaDeviceSynchronize();
	cudaMemcpy(pr, pk_gpu, nodes_number*sizeof(double), cudaMemcpyDeviceToHost);
	cout << pr[0] << endl;


	//cudaFree(pk);
	cudaFree(uniform_gpu);
	cudaFree(new_pk);
	cudaFree(pk_gpu);
	cudaFree(out);
	
	return 0;
}