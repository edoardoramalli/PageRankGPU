#include <iostream>
#include <stdio.h>
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <vector>
//#include <cub/cub.cuh>
//#include "Utilities.cuh"

using namespace std;

#define CONNECTIONS "data.csv"
#define BLOCKSIZE 256

//const int N = 256;


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

int main(){

   	ifstream connFile, probFile;
   	connFile.open(CONNECTIONS);
	vector<vector<double>> T_vector = {};
	int *row_ptrs;
	int *col_indices;
	int *connections;
	int nodes_number;
	int rows;
	int col_indices_number;
	int conn_size;

	cout << "Load connections" << endl;;
	if (connFile){
		string line, element;
		
		// Read rows number and allocate vector
		getline(connFile, line);
		rows = stoi(line);
		nodes_number = rows-1;
		row_ptrs = (int *) malloc(rows*sizeof(int));
		
		// Store menaningful rows
		getline(connFile, line);
		stringstream ss(line);
		for (int i = 0; i < rows; i++){
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
		cpu_sum += uniform_p;
	}

	double *pk_gpu, *out;
	cout << cpu_sum << endl;

	cudaMallocManaged(&pk_gpu, nodes_number*sizeof(double));
	//cudaMalloc(&pk, nodes_number*sizeof(double));
	cudaMallocManaged(&out, nodes_number*sizeof(double));
	for (int i = 0; i < nodes_number; i++) pk_gpu[i] = 1;
	//cout << "hi" << endl;
	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(double), cudaMemcpyHostToDevice);

	cuda_reduction <BLOCKSIZE> <<< nodes_number/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(pk_gpu, out, nodes_number);
	cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(double)>>>(out, pk_gpu, nodes_number);
	//addReduceKernel<<<nodes_number/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(pk_gpu, out);
	//addReduceKernel<<<1, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(out, pk_gpu);
	cudaDeviceSynchronize();
	cudaMemcpy(pr, pk_gpu, nodes_number*sizeof(double), cudaMemcpyDeviceToHost);
	cout << pr[0] << endl;


	//cudaFree(pk);
	cudaFree(pk_gpu);
	cudaFree(out);
	
	return 0;
}