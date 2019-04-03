#include <stdio.h>
#include <unistd.h>
#include "handleDataset.h"
#include <time.h>       /* time_t, time (for timestamp in second) */
#include <sys/timeb.h>  /* ftime, timeb (for timestamp in millisecond) */
#include "cuda_reduce.cu"

using namespace std;

#define DAMPING_F 0.85
#define THRESHOLD 0.000001



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void sauron_eye(float* pkCPU, float *old_pk, float *new_pk, int *empty_cols, int *row_indices, int *columns,
	float *data, float *damping, int *pk_len, int *data_len, int *empty_cols_len, int dampingFactor, float precisionThreshold){


	int block_number = (*pk_len + BLOCKSIZE - 1) / BLOCKSIZE;
	int uniform_blocks = (*empty_cols_len + BLOCKSIZE - 1)/BLOCKSIZE;
	int mul_blocks = (*data_len + BLOCKSIZE -1)/BLOCKSIZE;
	
	float *result, *out, *out_unif, *empty_contrib, *empty_value, *weighted, *thresholdGPU;
	bool *loop;
	cudaMalloc(&thresholdGPU, sizeof(float));
	cudaMalloc(&result, sizeof(float));
	cudaMallocManaged(&empty_contrib, sizeof(float));
	cudaMallocManaged(&loop, sizeof(bool));
	cudaMalloc(&out, sizeof(float)*block_number);
	cudaMalloc(&out_unif, sizeof(float)*block_number);
	cudaMallocManaged(&empty_value, sizeof(float));
	cudaMalloc(&weighted, *pk_len*sizeof(float));

	
	int i = 0;

	float * tmp;

	float teleportation = dampingFactor/ *pk_len;
	cudaMemcpy(empty_value, &teleportation, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(thresholdGPU, &precisionThreshold, sizeof(float), cudaMemcpyHostToDevice);

	
	*loop = true;

	// Get timestamp
	struct timeb timer_msec;
	long long int timestamp_start, timestamp_end; /* timestamp in millisecond. */
	
	while (*loop){
		//printf("-------------- Iteration %d ----------------\n", i);
		if (i!=0){
			tmp = old_pk;
			old_pk = new_pk;
			new_pk = tmp;
			cudaMemset(new_pk, 0, *pk_len*sizeof(float));
		}
		else{
			if (!ftime(&timer_msec)) {
				timestamp_start = ((long long int) timer_msec.time) * 1000ll + 
									(long long int) timer_msec.millitm;
			}
			else {
				timestamp_start = -1;
			}
		}


		uniform_reduction <BLOCKSIZE> <<<uniform_blocks, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (old_pk, empty_cols, out_unif, empty_value, *empty_cols_len);
		cuda_reduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>>(out_unif, empty_contrib, uniform_blocks);		

		pk_multiply<BLOCKSIZE> <<<mul_blocks, BLOCKSIZE>>>(data, columns, row_indices, old_pk, new_pk, *data_len, pk_len);


		*loop = false;


		sumAll<BLOCKSIZE> <<< block_number, BLOCKSIZE >>> (empty_contrib, damping, new_pk, pk_len);

		check_termination<1> <<<1, 1>>>(old_pk, new_pk, out, result, loop, pk_len, block_number, thresholdGPU);
		//printf("Check termination\n");

		i++;

		cudaDeviceSynchronize();
	}

	// Copy data back
	gpuErrchk(cudaMemcpy(pkCPU, new_pk, *pk_len * sizeof(float), cudaMemcpyDeviceToHost));

	if (!ftime(&timer_msec)) {
		timestamp_end = ((long long int) timer_msec.time) * 1000ll + 
							(long long int) timer_msec.millitm;
		}
	else {
	timestamp_end = -1;
	}

	cout << endl;
	cout << "Completed Convergence in " << i << " iterations" << endl;

	cout << "Time to convergence: " << (float)(timestamp_end - timestamp_start) / 1000 << endl;


	cudaFree(result);
	cudaFree(loop);
	cudaFree(out);
}


int main(int argc, char *argv[]){

	int nodes_number, col_indices_number, empty_len;
	float dampingMatrix;
	float dampingFactor = DAMPING_F;
	float precisionThreshold = THRESHOLD;

	string inputPath = "";
	string outputPath = "";

	int opt;
	while((opt = getopt(argc, argv, "i:o:sfd:t:"))!= EOF){
		switch (opt){
			case 'i':
				inputPath = optarg;
				outputPath =  "pk_" + inputPath; 
				break;
			case 'o':
				outputPath = optarg;
				break;
			case 's':
				inputPath = "data_small.csv";
				outputPath = "pk_data_small.csv";
				break;
			case 'f':
				inputPath = "data_full.csv";
				outputPath = "pk_data_full.csv";
				break;
			case 'd':
				dampingFactor = stof(optarg);
				break;
			case 't':
				precisionThreshold = stof(optarg);
				break;
			default:
				cout << "Invalid parameter " << opt << endl;
				exit(-1);
		}
	}


	if (inputPath == ""){
		cout << "Empty input path!" << endl;
		exit(-1);
	}

	if (outputPath == ""){
		cout << "Empty output path!" << endl;
		exit(-1);
	}

	if (precisionThreshold >= 1 | precisionThreshold < 0){
		cout << "Precision too coarse! Input a precision < 1" << endl;
		exit(-1);
	}

	if (dampingFactor >= 1 | dampingFactor < 0){
		cout << "Damping too big! Input a damping between 0 and 1" << endl;
		exit(-1);
	}

	cout << "Input dataset: " << inputPath << endl;
	cout << "Output file: " << outputPath << endl;
	cout << "Damping factor: " << dampingFactor << endl;
	cout << "Precision threshold: " << precisionThreshold << endl << endl;


	/*-----------------------------------------------------------------------*/

	loadDimensions(inputPath, nodes_number, col_indices_number, dampingMatrix, empty_len);

	int *row_ptrs = (int*) malloc(col_indices_number * sizeof(int));
	int *col_indices = (int*) malloc(col_indices_number * sizeof(int));
	int *empty_cols = (int*) malloc(empty_len * sizeof(int));
	float *connections = (float*) malloc(col_indices_number * sizeof(float));

	cout << "Allocated vectors succesfully!" << endl;
	
	loadDataset(inputPath, row_ptrs, col_indices, connections, empty_cols);

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

	cudaMalloc(&pk_gpu, nodes_number*sizeof(float));
	cudaMalloc(&new_pk, nodes_number*sizeof(float));
	cudaMalloc(&factor_gpu, sizeof(float));
	cudaMalloc(&c_gpu, col_indices_number*sizeof(int));
	cudaMalloc(&d_gpu, col_indices_number*sizeof(float));
	cudaMalloc(&r_gpu, col_indices_number*sizeof(int));
	cudaMallocManaged(&pk_len, sizeof(int));
	cudaMallocManaged(&data_len, sizeof(int));
	cudaMallocManaged(&empty_len_gpu, sizeof(int));
	cudaMalloc(&empty_gpu, empty_len*sizeof(int));

	// Populate device data from main memory

	cudaMemcpy(pk_gpu, pr, nodes_number*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(factor_gpu, &dampingMatrix, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu, col_indices, sizeof(int)*col_indices_number, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gpu, connections, sizeof(float)*col_indices_number, cudaMemcpyHostToDevice);
	cudaMemcpy(r_gpu, row_ptrs, sizeof(int)*col_indices_number, cudaMemcpyHostToDevice);	
	cudaMemcpy(pk_len, &nodes_number, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data_len, &col_indices_number, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(empty_len_gpu, &empty_len, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(empty_gpu, empty_cols, sizeof(int)*empty_len, cudaMemcpyHostToDevice);	
	
	sauron_eye(pr, pk_gpu, new_pk, empty_gpu, r_gpu, c_gpu, d_gpu, factor_gpu, pk_len, data_len, empty_len_gpu, dampingFactor, precisionThreshold);

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

	cout << endl;
	
	cout << "Writing output file..." << endl;

	storePagerank(pr, nodes_number, outputPath);

	cout << "Done!" << endl;
	
	return 0;
}