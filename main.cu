#include <stdio.h>
#include "handleDataset.h"
#include <time.h>       /* time_t, time (for timestamp in second) */
#include <sys/timeb.h>  /* ftime, timeb (for timestamp in millisecond) */
#include "cuda_reduce.cu"

//#include <cub/cub.cuh>
//#include "Utilities.cuh"

using namespace std;

#define CONNECTIONS "data_small.csv"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//sqrt(sum((y-x)**2))

void sauron_eye(float *old_pk, float *new_pk, int *empty_cols, int *row_indices, int *columns,
	float *data, float *damping, int *pk_len, int *data_len, int *empty_cols_len){


	// printf("P1 damping %.8f\n",*damping);
	// printf("P1 pk_len %d\n",*pk_len);
	// printf("P1 data_len %d\n",*data_len);


	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;
	int uniform_blocks = (*empty_cols_len + BLOCKSIZE - 1)/BLOCKSIZE;
	//printf("Block number: %d\n", block_number);
	
	float *result, *out, *out_unif, *uniform_contrib, *uniform_factor;
	bool *loop;
	cudaMalloc(&result, sizeof(float));
	cudaMalloc(&uniform_contrib, sizeof(float));
	cudaMallocManaged(&loop, sizeof(bool));
	cudaMalloc(&out, sizeof(float)*block_number);
	cudaMalloc(&out_unif, sizeof(float)*block_number);
	cudaMalloc(&uniform_factor, sizeof(float));
	
	int i = 0;

	float * tmp;

	float teleportation = DAMPING_F/ *pk_len;
	cudaMemcpy(uniform_factor, &teleportation, sizeof(float),cudaMemcpyHostToDevice);
	
	*loop = true;

	while (*loop){
		printf("-------------- Iteration %d ----------------\n", i);
		if (i!=0){
			tmp = old_pk;
			old_pk = new_pk;
			new_pk = tmp;
		}


		uniform_reduction <BLOCKSIZE> <<<uniform_blocks, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (old_pk, empty_cols, out_unif, uniform_factor, *empty_cols_len);
		cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>>(out_unif, uniform_contrib, uniform_blocks);
		printf("Uniform contribution calculation\n");

		handle_multiply<BLOCKSIZE> <<<block_number, BLOCKSIZE>>> (old_pk, new_pk, damping, row_indices, columns, data, uniform_contrib, *pk_len);
		printf("Multiply\n");

		*loop = false;

		check_termination<1> <<<1, 1>>>(old_pk, new_pk, out, result, loop, pk_len, block_number);
		printf("Check termination\n");

		i++;

		cudaDeviceSynchronize();
	}

	cudaFree(result);
	cudaFree(loop);
	cudaFree(out);

}

// 3.99458e-07
// 2.31906e-07
// 5.49669e-07


int main(){

    int nodes_number, col_indices_number, conn_size, row_len, empty_len;
	float damping;
	
	string datasetPath = CONNECTIONS;

	loadDimensions(datasetPath, nodes_number, col_indices_number, conn_size, row_len, damping, empty_len);

	int *row_ptrs = (int*) malloc(row_len * sizeof(int));
    int *col_indices = (int*) malloc(col_indices_number * sizeof(int));
    int *empty_cols = (int*) malloc(empty_len * sizeof(int));
	float *connections = (float*) malloc(conn_size * sizeof(float));

    cout << "Allocated vectors succesfully!" << endl;
	
	loadDataset(datasetPath, row_ptrs, col_indices, connections, empty_cols);

	cout << "Allocate and initialize PageRank" << endl;

	cout << "Nodes: " << nodes_number << endl;
	
	float *pr = (float*) malloc(nodes_number*sizeof(float));
	float uniform_p = 1/(float)nodes_number;
	// cout << "Uniform_p " << uniform_p << endl;
	for (int i = 0; i < nodes_number; i++){
		pr[i] = uniform_p;
	}

	cout << "Finished allocation" << endl;


	// GPU variables
	float *pk_gpu, *new_pk, *factor_gpu, *d_gpu;
	int *c_gpu, *r_gpu, *data_len, *pk_len, *empty_len_gpu, *empty_gpu;

	// Allocate device memory

	// int empty_columns = 1;
	// int empty_c[] = {2}; 

	cudaMalloc(&pk_gpu, nodes_number*sizeof(float));
	cudaMalloc(&new_pk, nodes_number*sizeof(float));
	cudaMalloc(&factor_gpu, sizeof(float));
	cudaMalloc(&c_gpu, col_indices_number*sizeof(int));
	cudaMalloc(&d_gpu, col_indices_number*sizeof(float));
	cudaMalloc(&r_gpu, (nodes_number+1)*sizeof(int));
	cudaMallocManaged(&pk_len, sizeof(int));
	cudaMallocManaged(&data_len, sizeof(int));
	cudaMallocManaged(&empty_len_gpu, sizeof(int));
	cudaMalloc(&empty_gpu, empty_len*sizeof(int));

	// Populate device data from main memory

	//cout << nodes_number << endl;
	cout << "DAMPING FROM CSV: " << damping << endl;

	//uniform_p = 1/3.0f;
	//float pr_test[] = {uniform_p, uniform_p, uniform_p};
	//float data_test[] = {0.5*0.85, 0.85, 0.5*0.85};
	// int col_test[] = {1,0,1};
	// int ptr_test[] = {0,1,1,3};
	// conn_size = 3;
	// col_indices_number = 3;
	// nodes_number = 3;
	//damping = 0.15/3;


	

	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(factor_gpu, &damping, sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(c_gpu, col_indices, sizeof(int)*col_indices_number, cudaMemcpyHostToDevice);
	// cudaMemcpy(c_gpu, col_indices, sizeof(int)*col_indices_number, cudaMemcpyHostToDevice);

	cudaMemcpy(d_gpu, connections, sizeof(float)*col_indices_number, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_gpu, data_test, sizeof(float)*col_indices_number, cudaMemcpyHostToDevice);

	cudaMemcpy(r_gpu, row_ptrs, sizeof(int)*(nodes_number+1), cudaMemcpyHostToDevice);
	
	// cudaMemcpy(r_gpu, ptr_test, sizeof(int)*(nodes_number+1), cudaMemcpyHostToDevice);
	
	cudaMemcpy(pk_len, &nodes_number, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data_len, &conn_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(empty_len_gpu, &empty_len, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(empty_gpu, empty_cols, sizeof(int)*empty_len, cudaMemcpyHostToDevice);	

	// Get timestamp
	struct timeb timer_msec;
	long long int timestamp_start, timestamp_end; /* timestamp in millisecond. */
	if (!ftime(&timer_msec)) {
	  timestamp_start = ((long long int) timer_msec.time) * 1000ll + 
						  (long long int) timer_msec.millitm;
	}
	else {
	  timestamp_start = -1;
	}

	
	sauron_eye(pk_gpu, new_pk, empty_gpu, r_gpu, c_gpu, d_gpu, factor_gpu, pk_len, data_len, empty_len_gpu);

	gpuErrchk( cudaDeviceSynchronize() );

	// Copy data back
	cudaMemcpy(pr, new_pk, nodes_number*sizeof(float), cudaMemcpyDeviceToHost);

	if (!ftime(&timer_msec)) {
		timestamp_end = ((long long int) timer_msec.time) * 1000ll + 
							(long long int) timer_msec.millitm;
		}
	else {
	timestamp_end = -1;
	}

	printf("--------Finished--------\n");

	cout << "Time to convergence: " << (float)(timestamp_end - timestamp_start) / 1000 << endl;

	cudaFree(new_pk);
	cudaFree(pk_gpu);
	cudaFree(r_gpu);
	cudaFree(c_gpu);
	cudaFree(d_gpu);
	cudaFree(factor_gpu);
	cudaFree(pk_len);
	cudaFree(data_len);
	cudaFree(empty_gpu);
	cudaFree(empty_len_gpu);

	for (int i = 0; i < 3; i++){
		cout << pr[i] << endl; 
	}

	storePagerank(pr, nodes_number, "pk_result_full.csv");
	
	return 0;
}