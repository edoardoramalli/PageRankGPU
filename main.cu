#include <iostream>
#include <stdio.h>
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

#define CONNECTIONS "data.csv"

__global__ void addReduceKernel(double *inData, double *outData, double *sharedData){
	//extern __shared__ double sharedData[];

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
	for (int i = 0; i < nodes_number; i++){
		pr[i] = 1/(double)nodes_number;
	}

	double *pk, *shared_pk, *out_sum;
	cout << pr[0] << endl;

	cudaMalloc(&shared_pk, nodes_number*sizeof(double));
	cudaMalloc(&pk, nodes_number*sizeof(double));
	cudaMalloc(&out_sum, nodes_number*sizeof(double));
	cudaMemcpy(pk, pr, nodes_number*sizeof(double), cudaMemcpyHostToDevice);

	addReduceKernel<<<nodes_number/2, 2>>>(shared_pk, out_sum, pk);
	
	cudaMemcpy(pr, pk, nodes_number*sizeof(float), cudaMemcpyDeviceToHost);
	cout << pr[0] << endl;


	cudaFree(pk);
	cudaFree(shared_pk);
	cudaFree(out_sum);
	
	return 0;
}