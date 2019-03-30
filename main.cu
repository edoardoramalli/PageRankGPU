#include <stdio.h>
#include "handleDataset.h"
#include <time.h>       /* time_t, time (for timestamp in second) */
#include <sys/timeb.h>  /* ftime, timeb (for timestamp in millisecond) */
#include "warpRed.cuh"

//#include <cub/cub.cuh>
//#include "Utilities.cuh"

using namespace std;

#define CONNECTIONS "data.csv"
#define BLOCKSIZE 2
#define PRECISION 1000000
#define THRESHOLD 0.000001
#define DAMPING_F 0.85

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Default reduction kernel from seminar slides - check for possible optimizations

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
		//warpReduce(sdata, tid);
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

template <unsigned int blockSize> __global__ void uniform_reduction(float *old_pk, int *empty_cols,float *reduct, float factor, size_t array_len) {
    extern volatile __shared__ float sdata[];
    
    size_t tid = threadIdx.x,
    gridSize = blockSize * gridDim.x,
    
    i = blockIdx.x * blockSize + tid;
    
	sdata[tid] = 0;
	//printf("factor for uniform reduction: %f\n", factor);
    
    while (i < array_len) {
		//printf("empty column: %d\n\n", empty_cols[i]);
        sdata[tid] += factor*old_pk[empty_cols[i]];
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
	if (tid == 0) {
		reduct[blockIdx.x] = sdata[0]*(*factor);
	}
}

// Perform first step of pagerank row by column product
template <unsigned int blockSize> __global__ void weighted_sum_partial(float *pagerank_in, float *reduct,
	int *column, float *mat_data, size_t row_len, size_t pk_len){
	extern volatile __shared__ float sdata[];
	size_t  tid = threadIdx.x, gridSize = blockSize *gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while (i < row_len) {
		//printf("----\nmat_data: %f\ncolumn: %d\nold_pagerank: %f\n------\n", mat_data[i], column[i], pagerank_in[column[i]]);
		sdata[tid] += mat_data[i]*pagerank_in[column[i]];
		//printf("result %d tid, %f\n" ,tid, sdata[tid]);
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
	int *row_indices, int *columns, float *mat_data, float uniform_factor,size_t pk_len){	
		
	int tid = blockIdx.x * blockSize + threadIdx.x;	
	if (tid < pk_len){
		new_pk[tid] = 0;
		//printf("tid: %d", tid);
		int index = row_indices[tid];  // Index of the first element of the row in columns array and data array
		int row_len = row_indices[tid+1] - index;

		// If there is data for the row...
		if (row_len > 0){
			float *mult, *result;
			int block_number = (row_len + BLOCKSIZE - 1) / BLOCKSIZE;
			cudaMalloc(&mult, sizeof(float)*block_number);
			cudaMalloc(&result, sizeof(float));
			//cudaDeviceSynchronize();

			weighted_sum_partial <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>>(old_pk, mult, &columns[index], &mat_data[index], row_len, pk_len);
			cudaDeviceSynchronize();
			cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>>(mult, result, block_number);
			
			cudaDeviceSynchronize();
			
			new_pk[tid] += *result;
			//printf("row %d partial result: %f\n", tid, *result);
			cudaFree(mult);
			cudaFree(result);
		}
		// Sum damping and uniform contribution
		new_pk[tid] += *damp + uniform_factor;
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

	termination_reduction <BLOCKSIZE> <<<block_number, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>> (new_pk, old_pk, out, *pk_len);
	cudaDeviceSynchronize();
	cuda_reduction <BLOCKSIZE> <<<1, BLOCKSIZE, BLOCKSIZE*sizeof(float) >>> (out, result, out_len);
	cudaDeviceSynchronize();
	float error;
	error = sqrtf(*result);
	printf("Error  %.10f\n", error);
	if (error - THRESHOLD > THRESHOLD) {
		*loop = true;
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
	cudaMallocManaged(&uniform_contrib, sizeof(float));
	cudaMallocManaged(&loop, sizeof(bool));
	cudaMalloc(&out, sizeof(float)*block_number);
	cudaMalloc(&out_unif, sizeof(float)*block_number);
	cudaMallocManaged(&uniform_factor, sizeof(float));
	
	int i = 0;

	float * tmp;

	*uniform_factor = DAMPING_F/ *pk_len;
	//printf("uniform factor: %f\n", uniform_factor);
	*loop = true;

	// for (int i = 0; i < 3; i++){
	// 	printf("%.4f\n", old_pk[i]);
	// }

	while (*loop){
		printf("-------------- Iteration %d ----------------\n", i);
		if (i!=0){
			tmp = old_pk;
			old_pk = new_pk;
			new_pk = tmp;
		}

		//printf("len :%d\n", *pk_len);
		// Calculate "damping contribution"
		//printf("Begin damping\n");
		// cuda_reduction <BLOCKSIZE> <<< block_number, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>>(old_pk, out, *pk_len);
		// cudaDeviceSynchronize();
		// damping_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (out, result, damping, block_number);
		// cudaDeviceSynchronize();

		//printf("Calculated damping: %.8f\n", *result);

		
		printf("Begin uniform contribution calculation\n");
		uniform_reduction <BLOCKSIZE> <<<uniform_blocks, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (old_pk, empty_cols, out_unif, *uniform_factor, *empty_cols_len);
		cudaDeviceSynchronize();
		cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>>(out_unif, uniform_contrib, uniform_blocks);
		cudaDeviceSynchronize();

		printf("Begin multiply\n");
		handle_multiply<BLOCKSIZE> <<<block_number, BLOCKSIZE>>> (old_pk, new_pk, damping, row_indices, columns, data, *uniform_contrib, *pk_len);
		cudaDeviceSynchronize();		
		*loop = false;
		printf("Begin check\n");
		check_termination<1> <<<1, 1>>>(old_pk, new_pk, out, result, loop, pk_len, block_number);

		cudaDeviceSynchronize();
		i++;
		// for (int i = 0; i < 3; i++){
		// 	printf("%.4f\n", new_pk[i]);
		// }
		//if (i == 3) break;
	}

	cudaFree(result);
	cudaFree(loop);
	cudaFree(out);

}


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
	
	float pr[nodes_number];
	float uniform_p = 1/(float)nodes_number;
	// cout << "Uniform_p " << uniform_p << endl;
	for (int i = 0; i < nodes_number; i++){
		pr[i] = uniform_p;
	}

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

	storePagerank(pr, nodes_number, "pk_result.csv");
	
	return 0;
}