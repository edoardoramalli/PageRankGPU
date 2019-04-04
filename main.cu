#include <stdio.h>
#include <unistd.h>
#include "handleDataset.h"
#include <time.h>       /* time_t, time (for timestamp in second) */
#include <sys/timeb.h>  /* ftime, timeb (for timestamp in millisecond) */
#include "cuda_reduce.cu"

using namespace std;

#define DAMPING_F 0.85
#define THRESHOLD 0.000001
#define SYNCHRONIZE 1 /* Remove definition to disable "extra" deviceSynchronize calls after kernel launch */



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
/* Check for eventual CUDA errors */
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void sauronEye(float* pkCPU, float *oldPkGPU, float *newPkGPU, int *emptyColIndices, int *rowIndicesGPU, int *columnIndicesGPU,
	float *matrixDataGPU, float *dampingMatrixFactorGPU, int *pkLenGPU, int *matrixDataLenGPU,
	 int *emptyColIndicesLenGPU, int dampingFactor, float precisionThreshold){

	/* Concealed within his fortress, the lord of Mordor sees all.
	His gaze pierces cloud, shadow, earth, and flesh.
	You know of what I speak, Gandalf: a great Eye, lidless, wreathed in flame. */

	int blockNumber = (*pkLenGPU + BLOCKSIZE - 1) / BLOCKSIZE;
	int uniformReductionBlocks = (*emptyColIndicesLenGPU + BLOCKSIZE - 1)/BLOCKSIZE;
	int mulBlocks = (*matrixDataLenGPU + BLOCKSIZE -1)/BLOCKSIZE;
	
	float *result, *out, *emptyColumnsContrib, *emptyColumnValue, *thresholdGPU;
	bool *loop;

	cudaMalloc(&thresholdGPU, sizeof(float));
	cudaMalloc(&result, sizeof(float));
	cudaMalloc(&emptyColumnsContrib, sizeof(float));
	cudaMallocManaged(&loop, sizeof(bool));
	cudaMalloc(&out, sizeof(float)*blockNumber);
	cudaMalloc(&emptyColumnValue, sizeof(float));

	
	int i = 0;

	float * tmp;

	float teleportation = dampingFactor/ *pkLenGPU;
	cudaMemcpy(emptyColumnValue, &teleportation, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(thresholdGPU, &precisionThreshold, sizeof(float), cudaMemcpyHostToDevice);

	
	*loop = true;

	// Get timestamp
	struct timeb timerMsec;
	long long int timestampStart, timestampEnd; /* timestamp in millisecond. */
	
	while (*loop){
		*loop = false;

		if (i!=0){
			// Swap pointers to avoid allocating new memory
			tmp = oldPkGPU;
			oldPkGPU = newPkGPU;
			newPkGPU = tmp;

			cudaMemset(newPkGPU, 0, *pkLenGPU*sizeof(float));	/* Set new pagerank to 0 */
		}
		else{
			// Get starting timestamp
			if (!ftime(&timerMsec)) {
				timestampStart = ((long long int) timerMsec.time) * 1000ll + 
									(long long int) timerMsec.millitm;
			}
			else {
				timestampStart = -1;
			}
		}

		uniformReduction <BLOCKSIZE> <<<uniformReductionBlocks, BLOCKSIZE, BLOCKSIZE *sizeof(float)>>> (oldPkGPU, emptyColIndices, out, emptyColumnValue, *emptyColIndicesLenGPU);
		cudaReduction <BLOCKSIZE> <<< 1, BLOCKSIZE, BLOCKSIZE*sizeof(float)>>>(out, emptyColumnsContrib, uniformReductionBlocks);		
		#ifdef SYNCHRONIZE
		cudaDeviceSynchronize();
		#endif

		pkMultiply<BLOCKSIZE> <<<mulBlocks, BLOCKSIZE>>>(matrixDataGPU, columnIndicesGPU, rowIndicesGPU, oldPkGPU, newPkGPU, *matrixDataLenGPU, pkLenGPU);
		#ifdef SYNCHRONIZE
		cudaDeviceSynchronize();
		#endif

		sumAll<BLOCKSIZE> <<< blockNumber, BLOCKSIZE >>> (emptyColumnsContrib, dampingMatrixFactorGPU, newPkGPU, pkLenGPU);

		checkTermination<1> <<<1, 1>>>(oldPkGPU, newPkGPU, out, result, loop, pkLenGPU, blockNumber, thresholdGPU);
		//printf("Check termination\n");

		i++;

		cudaDeviceSynchronize();
	}

	// Copy matrixDataGPU back
	gpuErrchk(cudaMemcpy(pkCPU, newPkGPU, *pkLenGPU * sizeof(float), cudaMemcpyDeviceToHost));

	// Get ending timestamp
	if (!ftime(&timerMsec)) {
		timestampEnd = ((long long int) timerMsec.time) * 1000ll + 
							(long long int) timerMsec.millitm;
		}
	else {
	timestampEnd = -1;
	}

	cout << endl;
	cout << "Completed Convergence in " << i << " iterations" << endl;

	cout << "Time to convergence: " << (float)(timestampEnd - timestampStart) / 1000 << endl;

	// Free allocated GPU resources

	cudaFree(thresholdGPU);
	cudaFree(result);
	cudaFree(emptyColumnsContrib);
	cudaFree(loop);
	cudaFree(out);
	cudaFree(emptyColumnValue);
}


int main(int argc, char *argv[]){

	int verticesNumber, colIndicesLen, emptyLen;
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
		cout << "Damping too big! Input a dampingMatrixFactorGPU between 0 and 1" << endl;
		exit(-1);
	}

	cout << "Input dataset: " << inputPath << endl;
	cout << "Output file: " << outputPath << endl;
	cout << "Damping factor: " << dampingFactor << endl;
	cout << "Precision threshold: " << precisionThreshold << endl << endl;


	/*-----------------------------------------------------------------------*/

	loadDimensions(inputPath, verticesNumber, colIndicesLen, dampingMatrix, emptyLen);

	cout << "Nodes: " << verticesNumber << endl;
	
	int *rowIndices = (int*) malloc(colIndicesLen * sizeof(int));
	int *colIndices = (int*) malloc(colIndicesLen * sizeof(int));
	int *emptyColIndices = (int*) malloc(emptyLen * sizeof(int));
	float *matrixData = (float*) malloc(colIndicesLen * sizeof(float));

	cout << "Allocated matrixDataGPU vectors succesfully!" << endl;
	
	loadDataset(inputPath, rowIndices, colIndices, matrixData, emptyColIndices);

	cout << "Allocate and initialize PageRank" << endl;

	
	float *pkCPU = (float*) malloc(verticesNumber*sizeof(float));
	float pkInit = 1/(float)verticesNumber;

	for (int i = 0; i < verticesNumber; i++){
		pkCPU[i] = pkInit;
	}

	cout << "Finished allocation" << endl;


	// GPU variables
	float *pkGPU, *newPkGPU, *dampingMatrixFactorGPU, *matrixDataGPU;
	int *columnIndicesGPU, *rowIndicesGPU, *matrixDataLenGPU, *pkLenGPU, *emptyColIndicesLenGPU, *emptyIndicesGPU;

	// Allocate device memory

	cudaMalloc(&pkGPU, verticesNumber*sizeof(float));
	cudaMalloc(&newPkGPU, verticesNumber*sizeof(float));
	cudaMalloc(&dampingMatrixFactorGPU, sizeof(float));
	cudaMalloc(&columnIndicesGPU, colIndicesLen*sizeof(int));
	cudaMalloc(&matrixDataGPU, colIndicesLen*sizeof(float));
	cudaMalloc(&rowIndicesGPU, colIndicesLen*sizeof(int));
	cudaMallocManaged(&pkLenGPU, sizeof(int));
	cudaMallocManaged(&matrixDataLenGPU, sizeof(int));
	cudaMallocManaged(&emptyColIndicesLenGPU, sizeof(int));
	cudaMalloc(&emptyIndicesGPU, emptyLen*sizeof(int));

	// Populate device matrixDataGPU from main memory

	cudaMemcpy(pkGPU, pkCPU, verticesNumber*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dampingMatrixFactorGPU, &dampingMatrix, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(columnIndicesGPU, colIndices, sizeof(int)*colIndicesLen, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixDataGPU, matrixData, sizeof(float)*colIndicesLen, cudaMemcpyHostToDevice);
	cudaMemcpy(rowIndicesGPU, rowIndices, sizeof(int)*colIndicesLen, cudaMemcpyHostToDevice);	
	cudaMemcpy(pkLenGPU, &verticesNumber, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matrixDataLenGPU, &colIndicesLen, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(emptyColIndicesLenGPU, &emptyLen, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(emptyIndicesGPU, emptyColIndices, sizeof(int)*emptyLen, cudaMemcpyHostToDevice);	

	// Start algorithm iteration
	
	sauronEye(pkCPU, pkGPU, newPkGPU, emptyIndicesGPU, rowIndicesGPU, columnIndicesGPU,
		 matrixDataGPU, dampingMatrixFactorGPU, pkLenGPU, matrixDataLenGPU,
		  emptyColIndicesLenGPU, dampingFactor, precisionThreshold);

	cudaFree(newPkGPU);
	cudaFree(pkGPU);
	cudaFree(rowIndicesGPU);
	cudaFree(columnIndicesGPU);
	cudaFree(matrixDataGPU);
	cudaFree(dampingMatrixFactorGPU);
	cudaFree(pkLenGPU);
	cudaFree(matrixDataLenGPU);
	cudaFree(emptyIndicesGPU);
	cudaFree(emptyColIndicesLenGPU);

	cout << endl;
	
	cout << "Writing output file..." << endl;

	storePagerank(pkCPU, verticesNumber, outputPath);

	cout << "Done!" << endl;
	
	return 0;
}