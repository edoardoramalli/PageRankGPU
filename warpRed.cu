// #include <iostream>

// using namespace std;


void deviceReduce(float *in, float* out, int N, int block_threads) {
    //int threads = 512;
	int blocks = min((N + block_threads - 1) / block_threads, 1024);
  
    deviceReduceWarpAtomicKernel<<<blocks, block_threads>>>(in, out, N);
    //deviceReduceWarpAtomicKernel<<<1, 1024>>>(out, out, blocks);
}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

// If all warps need the result

// __inline__ __device__
// float warpAllReduceSum(float val) {
//   for (int mask = warpSize/2; mask > 0; mask /= 2) 
//     val += __shfl_xor(val, mask);
//   return val;
// }

__global__ void deviceReduceWarpAtomicKernel(float *in, float* out, int N) {
    float sum = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < N; 
        i += blockDim.x * gridDim.x) {
      sum += in[i];
    }
    sum = warpReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0)
      atomicAdd(out, sum);
  }

// __inline__ __device__
// float blockReduceSum(float val) {

//   static __shared__ float shared[32]; // Shared mem for 32 partial sums
//   int lane = threadIdx.x % warpSize;
//   int wid = threadIdx.x / warpSize;

//   val = warpReduceSum(val);     // Each warp performs partial reduction

//   if (lane==0) shared[wid]=val; // Write reduced value to shared memory

//   __syncthreads();              // Wait for all partial reductions

//   //read from shared memory only if that warp existed
//   val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

//   if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

//   return val;
// }


// __global__ void deviceReduceKernel(int *in, int* out, int N) {
//     int sum = 0;
//     //reduce multiple elements per thread
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
//          i < N; 
//          i += blockDim.x * gridDim.x) {
//       sum += in[i];
//     }
//     sum = blockReduceSum(sum);
//     if (threadIdx.x==0)
//       out[blockIdx.x]=sum;
// }

// int main(){
// 	int *in, *out, N;
// 	N = 1500000;
// 	int *in_cpu, *out_cpu;
// 	in_cpu = (int*) malloc(N*sizeof(int));
// 	out_cpu = (int*) malloc(N*sizeof(int));
// 	cudaMalloc(&in, N*sizeof(int));
// 	cudaMalloc(&out, N*sizeof(int));
// 	for(int i = 0; i < N; i++){
// 		in_cpu[i] = 1;
// 	}
// 	cudaMemcpy(in, in_cpu, N*sizeof(int), cudaMemcpyHostToDevice);
//     int threads = 512;
// 	int blocks = min((N + threads - 1) / threads, 1024);
  
// 	deviceReduceWarpAtomicKernel<<<blocks, threads>>>(in, out, N);
// 	// cudaMemcpy(out_cpu, out, N*sizeof(int), cudaMemcpyDeviceToHost);
// 	// for(int i = 0; i < N; i++){
// 	// 	cout << out_cpu[i] << " ";
// 	// }
// 	// cout << endl;
// 	//deviceReduceWarpAtomicKernel<<<1, 1024>>>(out, out, blocks);	
// 	cudaDeviceSynchronize();
	
// 	cudaMemcpy(out_cpu, out, N*sizeof(int), cudaMemcpyDeviceToHost);

// 	cout << "result: " << out_cpu[0] << endl;
// 	return 0;

// }